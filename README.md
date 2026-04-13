# Voice AI Assistant

A real-time, browser-based voice AI assistant powered by **Amazon Nova Sonic** on AWS Bedrock.

Speak naturally and receive instant spoken responses — all through your browser.

---

## Quick Start

### Prerequisites

- **Node.js** >= 18.x
- **AWS Account** with Bedrock access enabled for Nova Sonic in `us-east-1`
- **IAM User** with `bedrock:InvokeModelWithBidirectionalStream` permission

### Setup

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your AWS credentials

# 3. Start the server
npm start

# 4. Open in browser
# Navigate to http://localhost:3000
# Use Chrome or Edge for best compatibility
```

---

## Architecture

```
Browser (Mic + Playback)
    ↕ WebSocket (binary audio + JSON control)
Node.js Server (Express + ws)
    ↕ Bedrock Bidirectional Stream
Amazon Nova Sonic (us-east-1)
```

---

## Browser Compatibility

| Browser | Support |
|---------|---------|
| Chrome 90+ | Full (Recommended) |
| Edge 90+ | Full (Recommended) |
| Firefox 85+ | Partial |
| Safari 15+ | Partial |

---

## License

Private — not licensed for redistribution.
