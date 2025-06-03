# Create an integration with Endpoints Metrics API

**This feature is currently in Beta. You will need to be subscribed to Enterprise to take advantage of this feature.**

You now have the ability to integrate the metrics of your Endpoint(s) to your internal tool. 

Utilizing OpenMetrics, you can create an integration to allow for a more granular view of your Endpoint's metrics in almost-real-time, showing for example:
- requests grouped by replica
- latency distribution of requests
- hardware metrics for all accelerator types

OpenMetrics is a standardized format for representing and transmitting time series data, making it easier for systems to consume and process metrics, ensuring that the data is structured optimally for storage and transport.

Further configurations and notifications can be set up for your Endpoints based on these metrics in your internal tool. 

## Connect with your internal tool

There are a variety of tools that work with OpenMetrics. You'll need to set up an agent. Here's some example docs to help get you started:

- [Datadog](https://docs.datadoghq.com/integrations/openmetrics/)
- [Grafana](https://grafana.com/docs/grafana-cloud/monitor-infrastructure/integrations/integration-reference/integration-metrics-endpoint/)

## Subscribe to Enterprise

You can sign up for an Enterprise plan starting at $20/user/mo at anytime at https://huggingface.co/enterprise?subscribe=true. 

For any questions or feature requests, please email us at api-enterprise@huggingface.co
