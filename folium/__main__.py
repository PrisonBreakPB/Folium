import sys


def main():
    if "--cli" in sys.argv:
        # remove --cli so argparse doesn't choke on it
        sys.argv.remove("--cli")
        from folium.cli import main as cli_main
        cli_main()
    else:
        from folium.web import run_server
        from folium.config import Config
        from folium.agent import Agent
        from folium.llm import LLM, LiteLLM

        config = Config.from_env()

        if not config.api_key:
            print("No API key found. Set OPENAI_API_KEY, DEEPSEEK_API_KEY, or FOLIUM_API_KEY")
            sys.exit(1)

        llm_cls = LiteLLM if config.provider == "litellm" else LLM
        llm = llm_cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_format=config.api_format,
        )
        agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)
        run_server(agent, config)


main()