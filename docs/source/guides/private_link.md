# Create a Private Endpoint with AWS PrivateLink

AWS PrivateLink enables you to privately connect your VPC to your Inference Endpoints, without exposing your traffic to the public
internet. It uses private IP addresses to route traffic between your VPC and the service, ensuring that data never traverses the
public internet, providing enhanced security and compliance benefits.

To create a Private Endpoint, you'll need to connect the AWS Account using the account ID. The following guide will walk you
through how to set it up.

## Configuring the AWS Private Link

### 1. Configure the Private Link

Under the "Security Level" setting you can toggle open the "AWS Private Link" section. The Private Link ensures the endpoint is only available through an intra-region secured AWS PrivateLink connection.

After providing your AWS Account ID and any other required information, click Create Endpoint. The endpoint creation process will begin.

![select private link](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_private_type.png)

After a few minutes, the endpoint will be created, and you will see the VPC Service Name in the overview. This name is necessary for
creating the VPC Interface Endpoint in your AWS account.

![vpc service name](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_3_vpc_ready.png)

### 2. Connect your VPC to your Interface Endpoint

Go to your AWS [console](https://console.aws.amazon.com/vpc/home?#Endpoints) and navigate to the VPC section to create the VPC Interface
Endpoint. Select "Other endpoint services" and enter the VPC Service Name provided earlier.

![add private link](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_4_add_private_link.png)

Verify the service name to ensure the connection is correct. Choose the VPC and subnets you wish to use for this endpoint. Make sure
they align with your security requirements.

![vpc endpoint](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_5_add_vpc.png)

### 3. Ready to Connect

After the VPC Endpoint status changes from pending to available, you should see an Endpoint URL in the overview. This URL can now
be used inside your VPC to access your endpoint in a secure and protected way, ensuring traffic is only occurring between the two
endpoints and will never leave AWS.

![endpoint running](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_6_running_endpoint.png)

## Shared Private Services 

If you have enabled the PrivateLink sharing option, you can now create additional endpoints that share the same VPC Endpoint. This
allows you to connect multiple endpoints to the same VPC Endpoint.

![shared private link](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/6_7_private_service_tooltip.png)

