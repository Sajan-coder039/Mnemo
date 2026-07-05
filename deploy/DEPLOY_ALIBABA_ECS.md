# Deploying Mnemo on Alibaba Cloud ECS

This guide runs the Mnemo backend (`server.py`, FastAPI) on an **Alibaba Cloud
ECS** instance. The backend calls Alibaba Cloud's **Qwen** models via the
**DashScope / Model Studio** endpoint (see `memoryagent/llm.py` and
`memoryagent/config.py`), so both the **compute** and the **AI service** run on
Alibaba Cloud.

```
Browser ──HTTP──▶ ECS (Docker: uvicorn + FastAPI) ──HTTPS──▶ Qwen / DashScope (Alibaba Cloud)
                          │
                          └── Chroma vector store on a mounted volume (/data)
```

---

## Prerequisites

1. An **Alibaba Cloud** account: https://www.alibabacloud.com
2. A **DashScope API key** (Model Studio): https://dashscope.console.aliyun.com
   → *API-KEY* → *Create*. Copy the `sk-...` value.
3. An SSH public key (yours is `~/.ssh/id_ed25519.pub`).

---

## Option A — Console + cloud-init (easiest)

### 1. Create the ECS instance
ECS console → **Instances** → **Create Instance**:

| Setting | Value |
|---|---|
| Billing | Pay-as-you-go (cheapest for a demo) |
| Region | Any near you (e.g. `ap-southeast-1` Singapore) |
| Instance type | `ecs.t6-c1m2.large` / `ecs.e-c1m2.large` (2 vCPU, 4 GB) — Chroma needs some RAM |
| Image | **Ubuntu 22.04 64-bit** |
| System disk | 40 GB ESSD |
| Public IP | **Assign public IPv4** (or bind an EIP) |
| Bandwidth | Pay-by-traffic, 5 Mbps |
| Key pair | Import/select your SSH key |

### 2. Paste the user-data script
Under **Advanced Settings → User Data**, paste the contents of
[`ecs-user-data.sh`](./ecs-user-data.sh) **after editing the
`DASHSCOPE_API_KEY` line**. On first boot it installs Docker, clones the repo,
builds the image, and starts the backend on port 80.

### 3. Open the firewall (Security Group)
Instance → **Security Groups** → inbound rules → **Add**:

| Port range | Source | Purpose |
|---|---|---|
| `80/80` | `0.0.0.0/0` | web UI + API |
| `22/22` | your IP | SSH |

### 4. Verify
Wait ~3–5 min for cloud-init, then open `http://<ECS-PUBLIC-IP>/` in a browser,
or:
```bash
curl -s http://<ECS-PUBLIC-IP>/api/status
# {"mode":"Qwen (qwen-plus)", ...}   ← "Qwen", not "offline stub", proves the
#                                       Alibaba Cloud model service is wired up
```

---

## Option B — SSH in and run the deploy script

If you skipped the user-data step (or want to redeploy):

```bash
ssh root@<ECS-PUBLIC-IP>          # uses your ~/.ssh/id_ed25519

git clone https://github.com/Sajan-coder039/Mnemo.git
cd Mnemo
DASHSCOPE_API_KEY=sk-xxxx ./deploy/deploy.sh
```

`deploy/deploy.sh` installs Docker if needed, builds the image, and runs the
container on port 80 with a persistent `mnemo-data` volume and
`--restart unless-stopped`.

---

## Option C — Alibaba Cloud CLI (`aliyun`) end-to-end

Install the CLI (`brew install aliyun-cli`), then `aliyun configure` with an
AccessKey. Create the instance and open the port from your laptop:

```bash
# 1. create a security group + open 80 and 22  (fill in <vpc-id>, <region>)
aliyun ecs CreateSecurityGroup --RegionId ap-southeast-1 --VpcId <vpc-id>
aliyun ecs AuthorizeSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupId <sg-id> --IpProtocol tcp --PortRange 80/80 --SourceCidrIp 0.0.0.0/0
aliyun ecs AuthorizeSecurityGroup --RegionId ap-southeast-1 \
  --SecurityGroupId <sg-id> --IpProtocol tcp --PortRange 22/22 --SourceCidrIp 0.0.0.0/0

# 2. run an instance (Ubuntu 22.04 image id varies by region; look one up with
#    aliyun ecs DescribeImages --RegionId ap-southeast-1 --OSType linux)
aliyun ecs RunInstances --RegionId ap-southeast-1 \
  --ImageId <ubuntu-2204-image-id> \
  --InstanceType ecs.e-c1m2.large \
  --SecurityGroupId <sg-id> --VSwitchId <vswitch-id> \
  --InternetMaxBandwidthOut 5 \
  --KeyPairName <your-key-pair> \
  --UserData "$(base64 -i deploy/ecs-user-data.sh)"
```

Then verify as in Option A step 4.

---

## Cost & teardown

A 2 vCPU / 4 GB pay-as-you-go instance is a few cents/hour. **Release the
instance** (ECS console → Instances → Release) when the demo/judging is over so
you stop being billed.

## Notes

- **Persistence:** Chroma + `memory_archive.jsonl` live in the `mnemo-data`
  Docker volume (`MEMORYAGENT_STORE=/data`), so memories survive restarts.
- **HTTPS:** for a custom domain with TLS, put an Alibaba Cloud SLB or Nginx +
  certbot in front of port 8000. Not required for the demo.
- Without `DASHSCOPE_API_KEY` the app still boots but runs the offline stub —
  set the key so `/api/status` reports `Qwen (qwen-plus)`.
