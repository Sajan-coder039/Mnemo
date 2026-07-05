#!/usr/bin/env bash
#
# One-shot: provision an Alibaba Cloud ECS instance and deploy Mnemo on it.
#
# Prereqs (run once):
#   brew install aliyun-cli jq
#   aliyun configure           # paste an AccessKey ID/Secret with ECS+VPC rights
#
# Usage:
#   DASHSCOPE_API_KEY=sk-xxxx ./deploy/provision_ecs.sh
#
# Optional overrides:
#   REGION=ap-southeast-1 INSTANCE_TYPE=ecs.e-c1m2.large PUBKEY=~/.ssh/id_ed25519.pub
#
# Creates (idempotent-ish): default VPC/vSwitch if missing, a security group
# (ports 22+80 open), imports your SSH key, finds the latest Ubuntu 22.04 image,
# and RunInstances with cloud-init user-data that builds + runs the container.
#
set -euo pipefail

REGION="${REGION:-ap-southeast-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ecs.e-c1m2.large}"
PUBKEY="${PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
SG_NAME="mnemo-sg"
KP_NAME="mnemo-key"
HERE="$(cd "$(dirname "$0")" && pwd)"

: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY (Alibaba Cloud DashScope key)}"
command -v aliyun >/dev/null || { echo "aliyun CLI missing (brew install aliyun-cli)"; exit 1; }
command -v jq     >/dev/null || { echo "jq missing (brew install jq)"; exit 1; }
[ -f "$PUBKEY" ]             || { echo "SSH public key not found: $PUBKEY"; exit 1; }

say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

say "Verifying credentials"
aliyun sts GetCallerIdentity >/dev/null || { echo "Run 'aliyun configure' first"; exit 1; }

say "Selecting a zone with stock for $INSTANCE_TYPE in $REGION"
ZONE=$(aliyun ecs DescribeAvailableResource --RegionId "$REGION" \
        --DestinationResource InstanceType --ResourceType instance \
        --InstanceChargeType PostPaid --InstanceType "$INSTANCE_TYPE" \
      | jq -r '.AvailableZones.AvailableZone[]? | select(.StatusCategory=="WithStock") | .ZoneId' | head -1)
[ -n "$ZONE" ] || { echo "No zone with stock for $INSTANCE_TYPE in $REGION. Try another REGION/INSTANCE_TYPE."; exit 1; }
echo "zone: $ZONE"

say "Ensuring a default VPC"
VPC=$(aliyun ecs DescribeVpcs --RegionId "$REGION" --IsDefault true | jq -r '.Vpcs.Vpc[0].VpcId // empty')
if [ -z "$VPC" ]; then
  VPC=$(aliyun vpc CreateDefaultVpc --RegionId "$REGION" | jq -r '.VpcId')
  sleep 5
fi
echo "vpc: $VPC"

say "Ensuring a vSwitch in $ZONE"
VSW=$(aliyun ecs DescribeVSwitches --RegionId "$REGION" --VpcId "$VPC" --ZoneId "$ZONE" \
      | jq -r '.VSwitches.VSwitch[0].VSwitchId // empty')
if [ -z "$VSW" ]; then
  VSW=$(aliyun vpc CreateDefaultVSwitch --RegionId "$REGION" --ZoneId "$ZONE" | jq -r '.VSwitchId')
  for _ in $(seq 1 20); do
    st=$(aliyun ecs DescribeVSwitches --RegionId "$REGION" --VpcId "$VPC" \
         | jq -r --arg v "$VSW" '.VSwitches.VSwitch[] | select(.VSwitchId==$v) | .Status')
    [ "$st" = "Available" ] && break; sleep 3
  done
fi
echo "vswitch: $VSW"

say "Ensuring security group $SG_NAME (open 22, 80)"
SG=$(aliyun ecs DescribeSecurityGroups --RegionId "$REGION" --SecurityGroupName "$SG_NAME" \
     | jq -r '.SecurityGroups.SecurityGroup[0].SecurityGroupId // empty')
if [ -z "$SG" ]; then
  SG=$(aliyun ecs CreateSecurityGroup --RegionId "$REGION" --VpcId "$VPC" --SecurityGroupName "$SG_NAME" | jq -r '.SecurityGroupId')
fi
for p in 22 80; do
  aliyun ecs AuthorizeSecurityGroup --RegionId "$REGION" --SecurityGroupId "$SG" \
    --IpProtocol tcp --PortRange "$p/$p" --SourceCidrIp 0.0.0.0/0 --Priority 1 >/dev/null 2>&1 || true
done
echo "sg: $SG"

say "Importing SSH key pair $KP_NAME"
aliyun ecs ImportKeyPair --RegionId "$REGION" --KeyPairName "$KP_NAME" \
  --PublicKeyBody "$(cat "$PUBKEY")" >/dev/null 2>&1 || echo "(key already imported)"

say "Finding latest Ubuntu 22.04 x86_64 system image"
IMG=$(aliyun ecs DescribeImages --RegionId "$REGION" --OSType linux --Architecture x86_64 \
        --ImageOwnerAlias system --PageSize 100 \
      | jq -r '[.Images.Image[].ImageId | select(startswith("ubuntu_22_04_x64"))] | sort | last')
[ -n "$IMG" ] && [ "$IMG" != "null" ] || { echo "No Ubuntu 22.04 image found in $REGION"; exit 1; }
echo "image: $IMG"

say "Rendering cloud-init user-data"
UD=$(sed "s|REPLACE_WITH_YOUR_ALIBABA_CLOUD_DASHSCOPE_KEY|$DASHSCOPE_API_KEY|" \
       "$HERE/ecs-user-data.sh" | base64 | tr -d '\n')

say "Launching instance ($INSTANCE_TYPE)"
IID=$(aliyun ecs RunInstances --RegionId "$REGION" \
        --ImageId "$IMG" --InstanceType "$INSTANCE_TYPE" \
        --SecurityGroupId "$SG" --VSwitchId "$VSW" \
        --InstanceName mnemo --HostName mnemo \
        --InternetChargeType PayByTraffic --InternetMaxBandwidthOut 5 \
        --KeyPairName "$KP_NAME" \
        --SystemDisk.Category cloud_essd --SystemDisk.Size 40 \
        --UserData "$UD" --Amount 1 \
      | jq -r '.InstanceIdSets.InstanceIdSet[0]')
echo "instance: $IID"

say "Waiting for public IP + Running state"
IP=""
for _ in $(seq 1 40); do
  J=$(aliyun ecs DescribeInstances --RegionId "$REGION" --InstanceIds "[\"$IID\"]")
  IP=$(echo "$J" | jq -r '.Instances.Instance[0].PublicIpAddress.IpAddress[0] // empty')
  ST=$(echo "$J" | jq -r '.Instances.Instance[0].Status // empty')
  [ -n "$IP" ] && [ "$ST" = "Running" ] && break
  sleep 6
done
[ -n "$IP" ] || { echo "No public IP yet; check ECS console for $IID"; exit 1; }

cat <<EOF

──────────────────────────────────────────────────────────
  Instance : $IID   ($INSTANCE_TYPE, $REGION/$ZONE)
  Public IP: $IP
  App URL  : http://$IP/
  SSH      : ssh root@$IP
──────────────────────────────────────────────────────────
cloud-init is now installing Docker + building the image (~3-6 min).
Poll until it answers:

  until curl -fs http://$IP/api/status; do sleep 10; done

Expect: {"mode":"Qwen (qwen-plus)", ...}   <- proves Alibaba Cloud Qwen is live

Teardown when done (stops billing):
  aliyun ecs DeleteInstance --RegionId $REGION --InstanceId $IID --Force true
EOF
