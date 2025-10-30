import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
import os

async def main():
    # Configure the browser and crawler
    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        # Content filtering
        word_count_threshold=10,
        excluded_tags=['form', 'header'],
        exclude_external_links=True,

        # Content processing
        process_iframes=True,
        remove_overlay_elements=True,

        # Cache control
        cache_mode=CacheMode.BYPASS  # Skip cache for fresh content
    )

    # Create and use the crawler
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.allianz.co.uk/insurance/car-insurance/existing-customers/claim.html#",
            config=run_config
        )

        if result.success:
            # Print clean content
            print("Content:", result.markdown[:500])  # First 500 chars

            # Define directory and file name
            save_dir = "../data"
            os.makedirs(save_dir, exist_ok=True)  # Create folder if not exists
            file_path = os.path.join(save_dir, "content.md")

            # Save the full markdown content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.markdown)

            print(f"Markdown saved at: {file_path}")

        else:
            print(f"Crawl failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())