# HuggingFace Setup Instructions

## Step 1: Accept Llama 3 License
1. Visit: https://huggingface.co/meta-llama/Meta-Llama-3-8B
2. Log in to your HuggingFace account (create one if needed)
3. Click the "Agree and access repository" button

## Step 2: Create HuggingFace Token
1. Visit: https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it something like "llm-second-order-effects"
4. Select token type: "Read" is sufficient for downloading models
5. Click "Generate token"
6. **COPY THE TOKEN** - you won't be able to see it again!

## Step 3: Login with the CLI
Run this command and paste your token when prompted:
```bash
source 2OE_env/bin/activate
huggingface-cli login
```

When asked "Add token as git credential?", you can choose "Y" for convenience.

## Step 4: Test the Setup
After logging in, run:
```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print('✓ Authentication successful!')"
```

## Next Steps
Once authenticated, you can proceed to download the models to the shared_models directory.
