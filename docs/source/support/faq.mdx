# FAQs 

## General questions

### In which regions can I deploy an Inference Endpoints?
Inference Endpoints are currently available on AWS in us-east-1 (N. Virginia) & eu-west-1 (Ireland), on Azure in eastus (Virginia), and on
GCP in us-east4 (Virginia). If you need to deploy in a different region, please let us know.

### Can I access the instance my Endpoint is running on?
No, you cannot access the instance hosting your Endpoint. But if you are missing information or need more insights on the machine where
the Endpoint is running, please contact us. 

### What's the difference between Inference Providers and Inference Endpoints? 
The [Inference Providers](https://huggingface.co/docs/inference-providers/index) is a solution to easily explore and evaluate models. Its a
single consistent API Inference giving access to Hugging Face partners, that host a wide selection of AI models. Inference Endpoints is a
service for you to deploy your models on managed infrastructure.

### How much does it cost to run my Endpoint?
Dedicated Endpoints are billed based on the compute hours of your Running Endpoints, and the associated instance types. We may add usage
costs for load balancers and Private Links in the future. 

### How do I monitor my deployed Endpoint?
You can currently monitor your Endpoint through the [Inference Endpoints web application](https://endpoints.huggingface.co/endpoints),
where you have access to the [Logs of your Endpoints](/docs/inference-endpoints/guides/logs) as well as a
[metrics dashboard](/docs/inference-endpoints/guides/analytics). 

## Security

### Is the data transiting to the Endpoint encrypted?
Yes, data is encrypted during transit with TLS/SSL.

### I accidentally leaked my token. Do I need to delete my endpoint?
You can invalidate existing personal tokens and create new ones in your settings here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
Please use fine-grained tokens when possible!

### Can I see my Private Endpoint running on my VPC account?
No, when creating a Private Endpoint (a Hugging Face Inference Endpoint linked to your VPC via AWS PrivateLink), you can only see the
ENI in your VPC where the Endpoint is available. 

## Configuration

### How can I scale my deployment?
The Endpoints are scaled automatically for you. You can set a minimum and maximum amount of replicas, and the system will scale them up and down
depending on the scaling strategy you configured. We recommend reading the [autoscaling section](./guides/autoscaling) for more information 

### Will my endpoint still be running if no more requests are processed?
Unless you allowed scale-to-zero your Inference Endpoint will always stay available/up with the number of min replicas defined in the Autoscaling
configuration 

### I would like to deploy a model which is not in the supported tasks, is this possible?
Yes, you can deploy any repository from the [Hugging Face Hub](https://huggingface.co/models) and if your task/model/framework is not
supported out of the box. For this we recommended setting up a [custom container](./engines/custom_container)

### What if I would like to deploy to a different instance type that is not listed?
Please contact us if you feel your model would do better on a different instance type than what is listed.

### I need to add a custom environment variable (default or secrets) to my endpoint. How can I do this?
This is now possible in the UI, or via the API:
```
{
  "model": {
    "image": {
      "huggingface": {
        "env": { "var1": "value" }
      }
    },
}
```

## Inference Engines

### Can I run inference in batches?
In most cases yes but it depends on the Inference Engine. In practice all high performance Inference Engines like vLLM, TGI, llama.cpp, SGLang
and TEI support batching, whereas the Inference Toolkit might not. Each Inference Enginge also has configuration to adjust batch sizes, we recommend
reading up on the documentation to understand best how to tune the configuration to meet your needs.

### I'm using a specific Inference Engine type for my Endpoint. Is there more information about how to use it? 
Yes! Please check the Inference Engines section and also check out the Engines own documentation.

## Debugging

### I can see from the logs that my endpoint is running but the status is stuck at "initializing"
This usually means that the port mapping is incorrect. Ensure your app is listening on port 80 and that the Docker container is exposing
port 80 externally. If you're deploying a custom container you can change these values, but make sure to keep them aligned.

### I'm getting a 500 response in the beginning of my endpoint deployment or when scaling is happening
Confirm that you have a health route implemented in your app that returns a status code 200 when your application is ready to serve
requests. Otherwise your app is considered ready as soon as the container has started, potentially resulting in 500s. You can configure
the health route in the Container Configuration of your Endpoint. 

You can also add the 'X-Scale-Up-Timeout' header to your requests. This means that when the endpoint is scaling the proxy will hold
requests until a replica is ready, or timeout after the specified amount of seconds. For example 'X-Scale-Up-Timeout: 600'

### I see there's an option to select a Download Pattern under Instance Configuration. What does this mean? 
You have an option to choose the download pattern of the model's files when deploying an Endpoint, to help with limiting the volume of
downloaded files. If a selected download pattern is not possible or compatible with the model, the system will not allow a change to the
pattern.

### I'm sometimes running into a 503 error on a running endpoint in production. What can I do? 
To help mitigate service interruptions on an Inference Endpoint that needs to be highly available, please make sure to use at least 2 replicas,
ie min replicas set to 2.

