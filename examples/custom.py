from strands import Agent, tool
@tool
def weather_forecast(city: str, days: int = 3) -> str:
    """Get weather forecast for a city.
    Args:
        city: The name of the city
        days: Number of days for the forecast
    """
    return f"Weather forecast for {city} for the next {days} days..."
agent = Agent(tools=[weather_forecast])
print(agent("What's the weather in Seattle tmw?"))
