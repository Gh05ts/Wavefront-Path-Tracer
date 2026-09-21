#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "protocol.hpp"

constexpr uint32_t distributedMaxFrameBytes = distributedMaxMessageBytes;

class TcpConnection {
public:
    TcpConnection() = default;
    ~TcpConnection();

    TcpConnection(const TcpConnection&) = delete;
    TcpConnection& operator=(const TcpConnection&) = delete;
    TcpConnection(TcpConnection&& other) noexcept;
    TcpConnection& operator=(TcpConnection&& other) noexcept;

    static std::optional<TcpConnection> connectTo(
        const std::string& host,
        uint16_t port,
        std::string* error = nullptr);

    bool send(const DistributedMessage& message, std::string* error = nullptr);
    bool receive(DistributedMessage& message, std::string* error = nullptr);
    bool valid() const { return socket_ >= 0; }
    void close();

private:
    friend class TcpListener;

    explicit TcpConnection(int socket) : socket_(socket) {}

    int socket_ = -1;
};

class TcpListener {
public:
    TcpListener() = default;
    ~TcpListener();

    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;
    TcpListener(TcpListener&& other) noexcept;
    TcpListener& operator=(TcpListener&& other) noexcept;

    static std::optional<TcpListener> listenOn(
        const std::string& host,
        uint16_t port,
        std::string* error = nullptr);

    std::optional<TcpConnection> accept(
        std::string* error = nullptr,
        uint32_t timeoutMs = 0xffffffffu);
    bool valid() const { return socket_ >= 0; }
    uint16_t port() const;
    void close();

private:
    explicit TcpListener(int socket) : socket_(socket) {}

    int socket_ = -1;
};
