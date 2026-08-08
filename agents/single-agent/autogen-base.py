"""currency tool-calling agent via create_agent."""

from __future__ import annotations
from typing import Literal
from langchain.agents import create_agent
from langchain.tools import tool
from pydantic import BaseModel, Field

from config import final_text, get_model_id


CurrencySymbol = Literal["USD", "EUR"]


def exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:
    if base_currency == quote_currency:
        return 1.0
    if base_currency == "USD" and quote_currency == "EUR":
        return 1 / 1.1
    if base_currency == "EUR" and quote_currency == "USD":
        return 1.1
    raise ValueError(f"unknown currencies {base_currency}, {quote_currency}")


class CurrencyCalcSchema(BaseModel):
    base_amount: float = Field(description="amount in base currency")
    base_currency: CurrencySymbol = Field(default="USD", description="base currency")
    quote_currency: CurrencySymbol = Field(default="EUR", description="quote currency")


@tool("currency_calculator", args_schema=CurrencyCalcSchema)
def currency_calculator(
    base_amount: float,
    base_currency: CurrencySymbol = "USD",
    quote_currency: CurrencySymbol = "EUR",
) -> str:
    """currency exchange calculator."""
    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount
    return f"{quote_amount:.4f} {quote_currency}"


def main() -> None:
    agent = create_agent(
        model=get_model_id(),
        tools=[currency_calculator],
        system_prompt=(
            "for currency exchange tasks, only use the tools you have been provided with. "
            "be concise."
        ),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "how much is 123.45 usd in eur?"}]}
    )
    print(final_text(result))


if __name__ == "__main__":
    main()
