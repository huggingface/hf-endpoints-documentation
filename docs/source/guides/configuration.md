# Configuration

This section describes the configuration options available when creating a new inference endpoint. Each section of
the interface allows fine-grained control over how the model is deployed, accessed, and scaled.

## Endpoint name, model and organization

In the top left you can:
- change the name of the inference endpoint
- verify to which organization you're deploying this model
- verify which model you are deploying
- and which Hugging Face Hub repo you are deploying this model from

![name-org-model](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/1-name-org-model.png)

## Hardware Configuration
The Hardware Configuration section allows you to choose the compute backend used to host the model.
You can select from three major cloud providers:
- Amazon Web Services (AWS)
- Microsoft Azure
- Google Cloud Platform

![hardware](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/2-hardware.png)

You must also choose an accelerator type:
- CPU
- GPU
- INF2 (AWS Inferentia)

Additionally, you can select the deployment region (e.g., East US) using the dropdown menu. Once the
provider, accelerator, and region are chosen, a list of available instance types is displayed. Each instance tile includes:

- GPU Type and Count
- Memory (e.g., 48 GB)
- vCPUs and RAM
- Hourly Pricing (e.g., $1.80 / h)

You can select a tile to choose that instance type for your deployment. Instances that are incompatible or unavailable in the
selected region are grayed out and unclickable.

## Authentication

This section determines who can access your deployed endpoint. Available options are:
- **Private (default)**: Accessible only to you, or members of your Hugging Face organization, using a personal HF access tokens.
- **Public**: Anyone can access your endpoint, without authentication.
- **HF Restricted**: Anyone with a Hugging Face account can access it, using their personal HF access tokens.

Additionally, if you deploy your Inference Endpoint in AWS, you can use **AWS PrivateLink** for an intra-region secured connection to your AWS VPN.

![auth](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/11-auth.png)

## Autoscaling

The Autoscaling section configures how many replicas of your model run and whether the system scales down to zero during periods of inactivity. For more
information we recommend reading the [in-depth guide on autoscaling](./autoscaling).

![autoscaling](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/4-autoscaling.png)

- **Automatic Scale-to-Zero**: A dropdown lets you choose how long the system should wait after the last request before
scaling down to zero. Default is after 1 hour with no activity.
- **Number of Replicas**:
    - Min: Minimum number of replicas to keep running. Note that enabling automatic scale-to-zero requires setting this to 0.
    - Max: Maximum number of replicas allowed (e.g., 1)
- **Autoscaling strategy**:
    - Based on hardware usage: For example, a scale up will be triggered if the average hardware utilisation (%) exceeds this threshold for more than 20 seconds.
    - Pending requests: A scale up event will be triggered if the average number of pending requests exceeds this threshold for more than 20 seconds.

## Inference Engine Configuration
This section allows you to specify how the container hosting your model behaves. This setting depends on the selected inference engine.
For configuration details, please read the Inference Engine section.
![inference-engine](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/9-inference-engine.png)

## Container Configuration
Here you can edit the container arguments and container command.
![container-configs](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/8-container-config.png)

## Environment Variables
Environment variables can be provided to customize container behavior or pass secrets.
- **Default Env**: Key-value pairs passed as plain environment variables.
- **Secret Env**: Key-value pairs stored securely and injected at runtime.

Each section allows you to add multiple entries using the Add button.

![env-vars](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/5-env-vars.png)

## Endpoint Tags
You can label endpoints with tags (e.g., for-testing) to help organize and manage deployments across environments or teams. In the dashboard
you will be able to filter and sort endpoints based on these tags.
Tags are plain text labels added via the Add button.

![tags](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/6-tags.png)

## Network
This section determines from where your deployed endpoint can be accessed. 

By default, your endpoint is accessible from the Internet, and secured with TLS/SSL. Endpoints deployed on an AWS instance can use AWS PrivateLink to restrict access to a specific VPC.

The available options are:
- Use AWS PrivateLink: check to activate AWS PrivateLink for your endpoint.
- AWS Account ID: You need to provide the AWS ID of the account that owns the VPC you want to restrict access to.
- PrivateLink Sharing: check to enable sharing of the same PrivateLink between different endpoints.

![network](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/10-network.png)

## Advanced Settings
Advanced Settings offer more fine-grained control over deployment.

![advanced](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/configuration/7-advanced.png)

- **Commit Revision**: Optionally specify a commit hash to which revision of the model repository on the Hugging Face Hub
you want to download the model artifacts from
- **Task**: Defines the type of model task. This is usually inferred from the model repository.
- **Container Arguments**: Pass CLI-style arguments to the container entrypoint.
- **Container Command**: Override the container entrypoint entirely.
- **Download Pattern**: Defines which model files are downloaded.