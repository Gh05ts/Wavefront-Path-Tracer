#include "distributed/transport.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <vector>

namespace
{
void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}

std::string systemError(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}

bool sendAll(int socket, const uint8_t* data, size_t byteCount, std::string* error) {
    size_t sent = 0;
    while (sent < byteCount) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const ssize_t result = ::send(socket, data + sent, byteCount - sent, flags);
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0) {
            setError(error, result == 0 ? "socket closed while sending" : systemError("send"));
            return false;
        }
        sent += static_cast<size_t>(result);
    }
    return true;
}

bool receiveAll(int socket, uint8_t* data, size_t byteCount, std::string* error) {
    size_t received = 0;
    while (received < byteCount) {
        const ssize_t result = ::recv(socket, data + received, byteCount - received, 0);
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0) {
            setError(error, result == 0 ? "socket closed while receiving" : systemError("recv"));
            return false;
        }
        received += static_cast<size_t>(result);
    }
    return true;
}
} // namespace

TcpConnection::~TcpConnection() {
    close();
}

TcpConnection::TcpConnection(TcpConnection&& other) noexcept
    : socket_(other.socket_) {
    other.socket_ = -1;
}

TcpConnection& TcpConnection::operator=(TcpConnection&& other) noexcept {
    if (this == &other)
        return *this;
    close();
    socket_ = other.socket_;
    other.socket_ = -1;
    return *this;
}

std::optional<TcpConnection> TcpConnection::connectTo(
    const std::string& host,
    uint16_t port,
    std::string* error) {
    if (host.empty()) {
        setError(error, "TCP connection host is empty");
        return std::nullopt;
    }

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;

    addrinfo* addresses = nullptr;
    const std::string service = std::to_string(port);
    const int lookup = ::getaddrinfo(host.c_str(), service.c_str(), &hints, &addresses);
    if (lookup != 0) {
        setError(error, std::string("getaddrinfo: ") + gai_strerror(lookup));
        return std::nullopt;
    }

    int connectedSocket = -1;
    std::string lastError = "could not connect to any resolved address";
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        const int socket = ::socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (socket < 0) {
            lastError = systemError("socket");
            continue;
        }
        if (::connect(socket, address->ai_addr, address->ai_addrlen) == 0) {
            connectedSocket = socket;
            break;
        }
        lastError = systemError("connect");
        ::close(socket);
    }
    ::freeaddrinfo(addresses);

    if (connectedSocket < 0) {
        setError(error, lastError);
        return std::nullopt;
    }
    return TcpConnection(connectedSocket);
}

bool TcpConnection::send(const DistributedMessage& message, std::string* error) {
    if (!valid()) {
        setError(error, "cannot send on an invalid TCP connection");
        return false;
    }

    std::vector<uint8_t> payload;
    if (!encodeDistributedMessage(message, payload, error))
        return false;
    if (payload.empty() || payload.size() > distributedMaxFrameBytes) {
        setError(error, "TCP message payload has an invalid size");
        return false;
    }

    const uint32_t networkLength = htonl(static_cast<uint32_t>(payload.size()));
    if (!sendAll(socket_, reinterpret_cast<const uint8_t*>(&networkLength), sizeof(networkLength), error))
        return false;
    return sendAll(socket_, payload.data(), payload.size(), error);
}

bool TcpConnection::receive(DistributedMessage& message, std::string* error) {
    if (!valid()) {
        setError(error, "cannot receive on an invalid TCP connection");
        return false;
    }

    uint32_t networkLength = 0;
    if (!receiveAll(socket_, reinterpret_cast<uint8_t*>(&networkLength), sizeof(networkLength), error))
        return false;
    const uint32_t length = ntohl(networkLength);
    if (length == 0 || length > distributedMaxFrameBytes) {
        setError(error, "TCP frame length exceeds the configured maximum");
        return false;
    }

    std::vector<uint8_t> payload(length);
    if (!receiveAll(socket_, payload.data(), payload.size(), error))
        return false;
    return decodeDistributedMessage(payload, message, error);
}

void TcpConnection::close() {
    if (socket_ >= 0) {
        ::close(socket_);
        socket_ = -1;
    }
}

TcpListener::~TcpListener() {
    close();
}

TcpListener::TcpListener(TcpListener&& other) noexcept
    : socket_(other.socket_) {
    other.socket_ = -1;
}

TcpListener& TcpListener::operator=(TcpListener&& other) noexcept {
    if (this == &other)
        return *this;
    close();
    socket_ = other.socket_;
    other.socket_ = -1;
    return *this;
}

std::optional<TcpListener> TcpListener::listenOn(
    const std::string& host,
    uint16_t port,
    std::string* error) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* addresses = nullptr;
    const std::string service = std::to_string(port);
    const char* hostValue = host.empty() ? nullptr : host.c_str();
    const int lookup = ::getaddrinfo(hostValue, service.c_str(), &hints, &addresses);
    if (lookup != 0) {
        setError(error, std::string("getaddrinfo: ") + gai_strerror(lookup));
        return std::nullopt;
    }

    int listeningSocket = -1;
    std::string lastError = "could not bind any resolved address";
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        const int socket = ::socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (socket < 0) {
            lastError = systemError("socket");
            continue;
        }

        int reuse = 1;
        ::setsockopt(socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (::bind(socket, address->ai_addr, address->ai_addrlen) == 0 && ::listen(socket, 64) == 0) {
            listeningSocket = socket;
            break;
        }
        lastError = systemError("bind/listen");
        ::close(socket);
    }
    ::freeaddrinfo(addresses);

    if (listeningSocket < 0) {
        setError(error, lastError);
        return std::nullopt;
    }
    return TcpListener(listeningSocket);
}

std::optional<TcpConnection> TcpListener::accept(std::string* error, uint32_t timeoutMs) {
    if (!valid()) {
        setError(error, "cannot accept on an invalid TCP listener");
        return std::nullopt;
    }

    if (timeoutMs != 0xffffffffu) {
        pollfd descriptor{};
        descriptor.fd = socket_;
        descriptor.events = POLLIN;
        const int pollResult = ::poll(&descriptor, 1, static_cast<int>(timeoutMs));
        if (pollResult == 0) {
            setError(error, "accept timed out");
            return std::nullopt;
        }
        if (pollResult < 0 && errno != EINTR) {
            setError(error, systemError("poll"));
            return std::nullopt;
        }
        if (pollResult < 0)
            return accept(error, timeoutMs);
    }

    int acceptedSocket = -1;
    do {
        acceptedSocket = ::accept(socket_, nullptr, nullptr);
    } while (acceptedSocket < 0 && errno == EINTR);

    if (acceptedSocket < 0) {
        setError(error, systemError("accept"));
        return std::nullopt;
    }
    return TcpConnection(acceptedSocket);
}

uint16_t TcpListener::port() const {
    if (!valid())
        return 0;

    sockaddr_storage address{};
    socklen_t addressLength = sizeof(address);
    if (::getsockname(socket_, reinterpret_cast<sockaddr*>(&address), &addressLength) != 0)
        return 0;
    if (address.ss_family == AF_INET)
        return ntohs(reinterpret_cast<const sockaddr_in*>(&address)->sin_port);
    if (address.ss_family == AF_INET6)
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&address)->sin6_port);
    return 0;
}

void TcpListener::close() {
    if (socket_ >= 0) {
        ::close(socket_);
        socket_ = -1;
    }
}
