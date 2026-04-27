# CTS-SAT-1-Ground-Station
Ground Station documentation and software for CTS-SAT-1 (FrontierSat)


# 01 - Router

Parker suggested to look into the following protocols:
* [Comparison Table of HTTP, MQTT and TCP](#comparison-table-of-differences)
* [HTTP (HyperText Transfer Protocol)](#http)
* [MQTT (Message Queuing Telemetry Transport)](#mqtt)
* [TCP (Transmission Control Protocol)](#tcp)

## Comparison Table of Differences

| Feature | MQTT | HTTP | TCP
|---|---|---|---|
| **Model** | Publish/Subscribe | Request/Response | Connection-oriented |
| **Ideal For**	| IoT & Low Bandwidth | Web & REST APIs | Data Streaming |
| **Overhead** | Very Low | High (Header) | Moderate (Transport) |
| **Reliability** | High (QoS) | High (via TCP) | High |
| **Payload** | Binary (Flexible) | Text (usually) | Raw Binary |

## HTTP
(HyperText Transfer Protocol)

When to use: Web browser communication, RESTful APIs, document transfer, and scenarios where devices only report data periodically (not real-time).

Why: Simple request-response model, easy to implement for REST services, and good for high-bandwidth, stable networks.

## MQTT
(Message Queuing Telemetry Transport)

When to use: IoT applications, sensor networks, battery-powered devices, unstable networks, or real-time messaging.

Why: Low overhead, publish/subscribe model reduces data consumption, and persistent connections allow for lightweight messaging. It is designed for low bandwidth.

## TCP
(Transmission Control Protocol)

When to use: Raw data transmission requiring guaranteed delivery, such as file transfers (FTP), email (SMTP), or custom high-reliability socket applications.

Why: It ensures reliable, ordered data transmission.
