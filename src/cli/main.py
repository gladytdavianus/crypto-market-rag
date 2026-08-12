import typer
from rich.console import Console

from src.generation.query_answer import answer_query
from src.generation.report_generator import generate_daily_report

app = typer.Typer(help="crypto-market-rag: manual query and report generation via terminal")
console = Console()


@app.command()
def query(question: str) -> None:
    """Ask a free-text question about the crypto market."""
    response = answer_query(question)

    console.print(f"\n[bold]{response.answer}[/bold]\n")
    console.print(f"Confidence: {response.confidence}")

    if response.coins_mentioned:
        console.print(f"Coins mentioned: {', '.join(response.coins_mentioned)}")

    if response.sources:
        console.print("\nSources:")
        for source in response.sources:
            console.print(f"  - ({source['type']}) {source['title']}")


@app.command()
def report() -> None:
    """Generate today's crypto market report."""
    result = generate_daily_report()

    console.print(f"\n[bold]Daily Report — {result.report_date.date()}[/bold]\n")
    console.print(result.summary)
    console.print(f"\nSentiment: {result.market_sentiment}")

    console.print("\nTop movers:")
    for mover in result.top_movers:
        sign = "+" if mover["change_pct"] >= 0 else ""
        console.print(f"  {mover['coin_id']}: {sign}{mover['change_pct']}%")


if __name__ == "__main__":
    app()
