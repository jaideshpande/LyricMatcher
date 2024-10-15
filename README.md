# LyricMatcher
Recommends songs based on the semantic similarity of the input song's lyrics. Alternatively, there's a search bar on the right side that lets you enter phrases or themes, such as "missing my ex" or "partying in the summer" and LyricMatcher will return the five songs with the highest Pinecone similarity score to that query.

I'm hosting this on Streamlit Cloud. Check it out here: https://lyricmatcher-jkojuiabcj7xstimzd2fq9.streamlit.app/

To run for yourself, you'll need to create your own API Keys for MUSIX_MATCH, OpenAI, Spotify Oauth, and Pinecone. Clone this repository and create a virtual python enviornment. Run 'pip install -r requirements.txt' to install all dependencies, and then set the API Keys in your code or as environment variables using export OPENAI_API_KEY='your_api_key'. Then do: streamlit run "appname.py" from inside that virtual environment.
