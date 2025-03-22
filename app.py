import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configure the Streamlit page
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")
st.title("Company News Sentiment Analyzer")
st.write("Enter a company name to analyze recent news sentiment and generate a Hindi audio summary.")

# Input field for the company name
company_name = st.text_input("Company Name", "Tesla")

if st.button("Analyze"):
    with st.spinner("Analyzing news articles..."):
        api_url = "http://localhost:8000/analyze-company"
        response = requests.post(api_url, json={"company_name": company_name})
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if we have valid results
            if data.get("fallback", False):
                st.warning("No recent news articles found. Showing default analysis.")
            
            st.subheader("Analysis Results")
            
            # Only show comparative analysis if available
            if "comparative_analysis" in data:
                st.write("## Comparative Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Handle sentiment distribution
                    sentiment_dist = data["comparative_analysis"].get("sentiment_distribution", {})
                    if sentiment_dist:
                        # Clean NaN/None values
                        clean_dist = {k: (v if v is not None else 0) for k, v in sentiment_dist.items()}
                        values = list(clean_dist.values())
                        labels = list(clean_dist.keys())
                        
                        if sum(values) > 0:
                            fig, ax = plt.subplots()
                            ax.pie(values, labels=labels, autopct='%1.1f%%')
                            ax.set_title("Sentiment Distribution")
                            st.pyplot(fig)
                        else:
                            st.write("No sentiment data available")
                    else:
                        st.write("No sentiment distribution data found")
                
                with col2:
                    # Handle common topics
                    st.write("### Common Topics")
                    if "common_topics" in data["comparative_analysis"]:
                        topics = data["comparative_analysis"]["common_topics"]
                        if topics:
                            topics_df = pd.DataFrame(topics, columns=["Topic", "Count"])
                            st.table(topics_df)
                        else:
                            st.write("No common topics identified")
                    else:
                        st.write("Topic analysis unavailable")
            
            # Display articles if available
            if "articles" in data and len(data["articles"]) > 0:
                st.write("## News Articles")
                for i, article in enumerate(data["articles"]):
                    with st.expander(f"{i+1}. {article.get('title', 'Untitled Article')}"):
                        st.write(f"**URL:** {article.get('url', 'No URL available')}")
                        st.write(f"**Summary:** {article.get('summary', 'No summary available')}")
                        st.write(f"**Sentiment:** {article.get('sentiment', {}).get('sentiment', 'Unknown')} "
                                f"(Score: {article.get('sentiment', {}).get('score', 0):.2f})")
                        st.write(f"**Topics:** {', '.join(article.get('topics', ['No topics identified']))}")
            else:
                st.write("## No Articles Found")
                st.info("Could not retrieve any news articles for analysis")
            
            # Display Hindi summary if available
            if "hindi_text" in data and data["hindi_text"]:
                st.write("## Hindi Summary")
                st.write(data["hindi_text"])
                
                if "speech_file" in data:
                    try:
                        with open(data["speech_file"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format="audio/mp3")
                    except FileNotFoundError:
                        st.error("Audio file not found")
            
            else:
                st.write("## Summary Unavailable")
                st.warning("Could not generate summary")
            
        else:
            st.error(f"Error: {response.text}")
