AWS CodeBuild for CUDA/GPU

Overview
- Use an ECR image based on an NVIDIA CUDA devel image with the Rust toolchain preinstalled. This repo provides `ci/docker/Dockerfile.cuda` as a starting point.
- Enable privileged mode so the build container can access GPUs via the host NVIDIA runtime.
- Choose a GPU-enabled compute type in your region (names vary; check AWS docs or your account limits). The example JSON uses `BUILD_GENERAL1_LARGE` as a placeholder.

Steps
1) Build and push the Docker image to ECR
   - aws ecr create-repository --repository-name scir-cuda-rust
   - docker build -t scir-cuda-rust:latest -f ci/docker/Dockerfile.cuda .
   - docker tag scir-cuda-rust:latest YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest
   - aws ecr get-login-password --region YOUR_REGION | docker login --username AWS --password-stdin YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com
   - docker push YOUR_AWS_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/scir-cuda-rust:latest

2) Create the CodeBuild project
   - Use `ci/codebuild/project.example.json` as a template; replace `YOUR_ORG`, account ID, region, and service role.
   - Ensure `privilegedMode` is true and your compute type supports GPU builds.
   - Set the buildspec to `buildspec.gpu.yml` in this repo.

3) Run the build
   - Provide source as this GitHub repo (or a connected CodeCommit repo).
   - Start the build; it will run `cargo build/test -p scir-gpu --features cuda`.
   - CUDA tests skip gracefully if GPU is not available.

Notes
- The container expects the CUDA Driver to be present on the host; CodeBuild’s GPU fleets provide that when GPU compute is enabled and privileged mode is on.
- For private GitHub repos, configure a Source Credential in CodeBuild or use a CodePipeline integration.

