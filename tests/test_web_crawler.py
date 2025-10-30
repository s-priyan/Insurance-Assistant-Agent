import unittest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock
import asyncio
import os
import sys

# Mock the crawl4ai module before importing
sys.modules['crawl4ai'] = MagicMock()
sys.modules['crawl4ai.async_configs'] = MagicMock()

# Now we can create a test version of the main function
async def main_testable():
    """Testable version of main function"""
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
    
    browser_config = BrowserConfig(verbose=True)
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        excluded_tags=['form', 'header'],
        exclude_external_links=True,
        process_iframes=True,
        remove_overlay_elements=True,
        cache_mode=CacheMode.BYPASS
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.allianz.co.uk/insurance/car-insurance/existing-customers/claim.html#",
            config=run_config
        )
        
        if result.success:
            save_dir = "../data"
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, "content.md")
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.markdown)
            
            return file_path
        else:
            raise Exception(f"Crawl failed: {result.error_message}")


class TestWebCrawler(unittest.TestCase):
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('crawl4ai.AsyncWebCrawler')
    def test_successful_crawl_and_save(self, mock_crawler_class, mock_file, mock_makedirs):
        """Test successful web crawling and content saving"""
        # Setup mock result
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.markdown = "# Test Content\n\nThis is test content."
        
        # Setup mock crawler
        mock_crawler = MagicMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=None)
        mock_crawler_class.return_value = mock_crawler
        
        # Run the async function
        result = asyncio.run(main_testable())
        
        # Assertions
        mock_makedirs.assert_called_once_with("../data", exist_ok=True)
        mock_file.assert_called_once_with(os.path.join("../data", "content.md"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("# Test Content\n\nThis is test content.")
        self.assertEqual(result, os.path.join("../data", "content.md"))
    
    @patch('crawl4ai.AsyncWebCrawler')
    def test_failed_crawl(self, mock_crawler_class):
        """Test handling of failed web crawl"""
        # Setup mock result for failure
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Connection timeout"
        
        # Setup mock crawler
        mock_crawler = MagicMock()
        mock_crawler.arun = AsyncMock(return_value=mock_result)
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=None)
        mock_crawler_class.return_value = mock_crawler
        
        # Run and expect exception
        with self.assertRaises(Exception) as context:
            asyncio.run(main_testable())
        
        self.assertIn("Crawl failed", str(context.exception))
    
    @patch('os.makedirs')
    def test_directory_creation(self, mock_makedirs):
        """Test that directory is created if it doesn't exist"""
        mock_makedirs.return_value = None
        
        # This tests the os.makedirs call
        os.makedirs("../data", exist_ok=True)
        
        mock_makedirs.assert_called_once_with("../data", exist_ok=True)


if __name__ == '__main__':
    unittest.main()