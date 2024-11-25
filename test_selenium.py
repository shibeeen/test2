from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

# Set up WebDriver (this example uses Chrome)
driver = webdriver.Chrome()

# Open the local host or deployed app where the model is running (replace with actual URL)
driver.get("http://localhost:5000")  # Example URL

# Find input fields (assuming the web page has input fields for features)
shopping_frequency = driver.find_element("name", "shopping_frequency")
age_group = driver.find_element("name", "age_group")
electronics_platform = driver.find_element("name", "electronics_platform")
fashion_platform = driver.find_element("name", "fashion_platform")

# Fill out the form with test data (replace with actual form field names and values)
shopping_frequency.send_keys("Weekly")
age_group.send_keys("25-34")
electronics_platform.send_keys("Amazon")
fashion_platform.send_keys("Myntra")

# Submit the form (assuming there is a submit button)
submit_button = driver.find_element("name", "submit")
submit_button.click()

# Wait for the results page to load (adjust time as needed)
time.sleep(3)

# Verify the results (this could be checking specific outputs on the results page)
result = driver.find_element("id", "result")  # Example: replace with actual result element ID
print("Prediction Result:", result.text)

# Close the browser
driver.quit()
