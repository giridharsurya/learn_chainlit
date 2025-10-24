from pydantic import BaseModel, Field
import requests
import mcp.types as types
from typing import Annotated
from semantic_kernel.functions import kernel_function
from typing import TypedDict as typed_dict
import os


class WeatherRequest(typed_dict):
    city: Annotated[str,"city name"]

class WeatherResponse(typed_dict):
    city: str
    temp_c: float
    condition: str
    wind_kph: float

class ForecastRequest(typed_dict):
    city: Annotated[str,"city name"]
    future_no_of_days: Annotated[int,"number of days for forecast, between 1 and 5"]

class ForecastResponse(typed_dict):
    city: str
    forecast: list[dict]

class Weather:

    @kernel_function(
        name="get_weather",
        description="Fetches weather information for a given city.",
    )
    def get_weather(self,city:  Annotated[str,"city name"]) -> str:
        api_key = os.environ['weather_api_key']  # Replace with your actual API key
        base_url = "http://api.weatherapi.com/v1/current.json" 
        params = {"key": api_key, "q": city}
        response = requests.get(base_url, params=params)    
        if response.status_code == 200:
            result = response.json()
            weather_response = WeatherResponse(
                city=result["location"]["name"], temp_c=result["current"]["temp_c"],
                condition=result["current"]["condition"]["text"],
                wind_kph=result["current"]["wind_kph"])
            return str(weather_response)
        else:
            error_response = response.json()
            return str(error_response)
        
    @kernel_function(
        name="forecast_weather",
        description="Fetches weather forecast for a given city and number of days.",
    )  
    def forecast_weather(self,city: Annotated[str,"city name"], future_no_of_days: Annotated[int,"number of days for forecast, between 1 and 5"]) -> str:
        api_key = os.environ['weather_api_key']  # Replace with your actual API key
        base_url = "http://api.weatherapi.com/v1/forecast.json" 
        params = {"key": api_key, "q": city,"days": future_no_of_days}
        response = requests.get(base_url, params=params)    
        if response.status_code == 200:
            result = response.json()
            forecast_data = []
            for day in result["forecast"]["forecastday"]:
                forecast_data.append({
                    "date": day["date"],
                    "max_temp_c": day["day"]["maxtemp_c"],
                    "min_temp_c": day["day"]["mintemp_c"],
                    "condition": day["day"]["condition"]["text"]
                })
            forecast_response = ForecastResponse(
                city=result["location"]["name"],
                forecast=forecast_data
            )
            return str(forecast_response)
        else:
            error_response = response.json()
            return str(error_response)
