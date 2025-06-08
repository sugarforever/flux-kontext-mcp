# Flux.1 Kontext MCP Server

MCP server configuration:

```json
{
    "mcpServers": {
        "flux_kontext_mcp": {
            "command": "uvx",
            "args": ["flux-kontext-mcp"],
            "env": {
                "REPLICATE_API_TOKEN": "your_replicate_api_token"
            }
        }
    }
}
```

## MCP tools

- flux_kontext_generate
- flux_kontext_async_generate
- flux_kontext_get_prediction
- flux_kontext_list_predictions
- flux_kontext_cancel_prediction
- flux_kontext_wait_for_prediction
- flux_kontext_get_model_info



