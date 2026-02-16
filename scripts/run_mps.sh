#!/bin/bash
# Quick launcher for M4 Pro training

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GRP-Oblit M4 Pro Launcher ===${NC}\n"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}⚠  OPENAI_API_KEY not set${NC}"
    echo "Set it with: export OPENAI_API_KEY=sk-..."
    echo "Or create .env file in project root"
    echo ""
fi

# Show options
echo "Select experiment:"
echo "  1) dev-mps      (~2-3 min)  - Ultra-fast iteration (fp16, 28GB)"
echo "  2) oblit-1-mps  (~5-15 min) - Full Oblit-1 reproduction (fp16, 28GB)"
echo "  3) smoke-test   (~30 sec)   - Verify setup only"
echo ""
read -p "Choice [1-3]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running dev-mps experiment with fp16...${NC}"
        uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
        ;;
    2)
        echo -e "\n${GREEN}Running oblit-1-mps experiment with fp16...${NC}"
        uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps
        ;;
    3)
        echo -e "\n${GREEN}Running smoke test...${NC}"
        uv run python scripts/smoke_test.py
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✓ Done!${NC}"
