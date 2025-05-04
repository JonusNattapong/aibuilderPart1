import streamlit as st
import os
import json
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import logging
from config_apigen_llm import (
    API_LIBRARY, NUM_GENERATIONS_TO_ATTEMPT, GENERATION_TEMPERATURE,
    ENABLE_FORMAT_CHECK, ENABLE_EXECUTION_CHECK, ENABLE_SEMANTIC_CHECK,
    USE_THAI_QUERIES, SAVE_CSV_OUTPUT, GENERATE_VISUALIZATIONS,
    SIMULATED_OUTPUT_DIR
)
from run_apigen_simulation import run_pipeline
from util_apigen import load_jsonl

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create output directory if it doesn't exist
os.makedirs(SIMULATED_OUTPUT_DIR, exist_ok=True)

def main():
    st.set_page_config(page_title="DatasetAPI Generator", layout="wide")
    st.title("DatasetAPI Generator")

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("DEEPSEEK API Key", type="password")
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    # Generation settings
    st.sidebar.subheader("Generation Settings")
    num_generations = st.sidebar.slider("Number of Examples", 1, 100, NUM_GENERATIONS_TO_ATTEMPT)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, GENERATION_TEMPERATURE)
    use_thai = st.sidebar.checkbox("Generate Thai Queries", USE_THAI_QUERIES)

    # Validation settings
    st.sidebar.subheader("Validation Settings")
    format_check = st.sidebar.checkbox("Format Check", ENABLE_FORMAT_CHECK)
    execution_check = st.sidebar.checkbox("Execution Check", ENABLE_EXECUTION_CHECK)
    semantic_check = st.sidebar.checkbox("Semantic Check", ENABLE_SEMANTIC_CHECK)

    # Output settings
    st.sidebar.subheader("Output Settings")
    save_csv = st.sidebar.checkbox("Save CSV", SAVE_CSV_OUTPUT)
    generate_viz = st.sidebar.checkbox("Generate Visualizations", GENERATE_VISUALIZATIONS)

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Generator", "Results", "Visualization"])

    with tab1:
        st.header("Dataset Generator")
        
        # Display available APIs
        st.subheader("Available APIs")
        api_df = []
        for name, details in API_LIBRARY.items():
            api_df.append({
                'API': name,
                'Description': details['description'],
                'Category': details.get('category', 'Other')
            })
        st.dataframe(pd.DataFrame(api_df))

        # Generation section
        col1, col2 = st.columns([3, 1])
        with col1:
            generate_button = st.button("Generate Dataset", use_container_width=True)
        with col2:
            st.metric("Examples to Generate", num_generations)

        if generate_button:
            if not api_key:
                st.error("Please enter your DEEPSEEK API Key in the sidebar")
                return

            # Update configuration
            import config_apigen_llm as config
            config.NUM_GENERATIONS_TO_ATTEMPT = num_generations
            config.GENERATION_TEMPERATURE = temperature
            config.USE_THAI_QUERIES = use_thai
            config.ENABLE_FORMAT_CHECK = format_check
            config.ENABLE_EXECUTION_CHECK = execution_check
            config.ENABLE_SEMANTIC_CHECK = semantic_check
            config.SAVE_CSV_OUTPUT = save_csv
            config.GENERATE_VISUALIZATIONS = generate_viz

            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                col1, col2 = st.columns(2)
                with col1:
                    stage_text = st.empty()
                with col2:
                    progress_text = st.empty()

                def update_progress(progress: float):
                    progress_bar.progress(progress)
                    progress_text.text(f"Progress: {progress:.1%}")

                def update_stage(stage: str):
                    stage_text.text(f"Stage: {stage}")

                # Run generation with progress tracking
                try:
                    with st.spinner("Running pipeline..."):
                        run_pipeline(
                            progress_callback=update_progress,
                            stage_callback=update_stage
                        )
                        progress_bar.progress(1.0)
                        st.success("Dataset generation completed!")
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
                    logger.exception("Pipeline error")
                finally:
                    # Keep the final progress display
                    pass

            # Add expander for detailed stats after completion
            with st.expander("Generation Statistics", expanded=True):
                try:
                    stats_path = os.path.join(SIMULATED_OUTPUT_DIR, "api_call_statistics.json")
                    if os.path.exists(stats_path):
                        with open(stats_path, 'r') as f:
                            stats = json.load(f)
                        
                        # Display key metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Generated", stats.get("total_attempted", 0))
                        with col2:
                            st.metric("Passed", stats.get("pass", 0))
                        with col3:
                            st.metric("Failed",
                                    stats.get("fail_format", 0) +
                                    stats.get("fail_execution", 0) +
                                    stats.get("fail_semantic", 0))
                        with col4:
                            st.metric("Needs Review", stats.get("needs_review", 0))
                except FileNotFoundError:
                    st.info("Statistics not available yet. Generate a dataset first.")
                except json.JSONDecodeError as e:
                    st.error(f"Error reading statistics file: {str(e)}")
                except Exception as e:
                    logger.exception("Error loading statistics")
                    st.error(f"Unexpected error loading statistics: {str(e)}")

    with tab2:
        st.header("Generated Results")
        
        # Load and display results if available
        try:
            results_path = "DataOutput/apigen_llm_diverse/verified_api_calls_llm_diverse.jsonl"
            if os.path.exists(results_path):
                results = load_jsonl(results_path)
                
                # Display results as table
                results_df = []
                for item in results:
                    query = item.get('query', '')
                    for result in item.get('execution_results', []):
                        call = result.get('call', {})
                        results_df.append({
                            'Query': query,
                            'API': call.get('name', ''),
                            'Arguments': json.dumps(call.get('arguments', {})),
                            'Success': result.get('execution_success', False),
                            'Output': json.dumps(result.get('execution_output', {}))
                        })
                
                if results_df:
                    st.dataframe(pd.DataFrame(results_df))
                else:
                    st.info("No results found in the dataset")
            else:
                st.info("No generated dataset found. Use the Generator tab to create one.")
        except FileNotFoundError:
            st.info("No dataset found. Use the Generator tab to create one.")
        except json.JSONDecodeError as e:
            st.error(f"Error reading dataset file: {str(e)}")
        except Exception as e:
            logger.exception("Error loading results")
            st.error(f"Unexpected error loading results: {str(e)}")

    with tab3:
        st.header("Visualizations")
        
        try:
            # Load statistics
            stats_path = "DataOutput/apigen_llm_diverse/api_call_statistics.json"
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)

                # API Distribution
                if 'api_call_counts' in stats:
                    st.subheader("API Call Distribution")
                    api_df = pd.DataFrame(list(stats['api_call_counts'].items()), 
                                        columns=['API', 'Count'])
                    fig = px.pie(api_df, values='Count', names='API', 
                               title='Distribution of API Calls')
                    st.plotly_chart(fig)

                # Query Length Distribution
                if 'query_lengths' in stats and 'distribution' in stats['query_lengths']:
                    st.subheader("Query Length Distribution")
                    query_df = pd.DataFrame(list(stats['query_lengths']['distribution'].items()),
                                         columns=['Length Range', 'Count'])
                    fig = px.bar(query_df, x='Length Range', y='Count',
                                title='Distribution of Query Lengths')
                    st.plotly_chart(fig)

                # Success/Failure Statistics
                st.subheader("Generation Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Attempted", stats.get('total_attempted', 0))
                with col2:
                    st.metric("Passed", stats.get('pass', 0))
                with col3:
                    st.metric("Needs Review", stats.get('needs_review', 0))

            else:
                st.info("No statistics found. Generate a dataset to view visualizations.")
        except FileNotFoundError:
            st.info("No visualizations available. Generate a dataset first.")
        except json.JSONDecodeError as e:
            st.error(f"Error reading visualization data: {str(e)}")
        except Exception as e:
            logger.exception("Error loading visualizations")
            st.error(f"Unexpected error loading visualizations: {str(e)}")

if __name__ == "__main__":
    main()