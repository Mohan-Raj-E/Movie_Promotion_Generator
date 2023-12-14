import streamlit as st
# Importing necessary libraries
import os
import re
from pathlib import Path
from datetime import datetime, timedelta
import traceback

# Data manipulation and analysis
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import statistics


def get_view_category(view_count):
    """
    Returns the view count category based on the given view count.

    Args:
        view_count (int): The view count.

    Returns:
        str: The view count category.
    """
    try:
        if view_count >= 1000000:
            return f"{view_count / 1000000:.1f} Million"
        elif view_count >= 100000:
            return f"{view_count / 100000:.1f} Lakh"
        elif view_count >= 1000:
            return f"{view_count / 1000:.1f} K"
        else:
            return str(view_count)
    except Exception as e:
        st.error("An error occurred while determining the view count category.")
        st.exception(e)


def calculate_cumulative_statistics(data, modified_movie_name):
    try:
        # Initialize variables for cumulative statistics
        total_sum_views = 0
        total_sum_likes = 0
        total_sum_comments = 0
        total_engagement_ratio = 0
        file_count = 0

        # Calculate sum of the "Views," "Likes," and "Comments" columns
        sum_views = data["Views"].sum()
        sum_likes = data["Likes"].sum()
        sum_comments = data["Comments"].sum()

        # Accumulate the values
        total_sum_views += sum_views
        total_sum_likes += sum_likes
        total_sum_comments += sum_comments

        # Calculate engagement ratio for each file
        engagement_ratio = (sum_likes + sum_comments) / sum_views * 100
        total_engagement_ratio += engagement_ratio

        # Display the results in Streamlit
        st.write(f"**Movie Name:** **{modified_movie_name.title()}**")

        st.write(f"Total Views: {get_yt_category(sum_views)}")
        st.write(f"Total Likes: {get_yt_category(sum_likes)}")
        st.write(f"Total Comments: {get_yt_category(sum_comments)}")
        st.write(f"Engagement Ratio: {engagement_ratio:.2f}%")
        st.write("-------------------------------------")

    except Exception as e:
        st.error("An error occurred while determining the calculate_cumulative_statistics.")
        st.exception(e)


def get_order_of_promotional_content_overall(full_subfolder_path, files):
    # Dictionary to store the order of appearance and views for each classification value
    classification_order_views = defaultdict(list)

    combined_df = pd.DataFrame()

    # List to store filtered DataFrames
    filtered_dfs = []

    # Iterate over each file
    for file in files:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            # Construct the full file path
            file_path = os.path.join(full_subfolder_path, file)

            try:
                # Read the Excel file into a pandas DataFrame
                df = pd.read_excel(file_path)
                # Convert 'Published_date' and 'Release_Date' columns to datetime
                df['Published_date'] = pd.to_datetime(df['Published_date'])
                df['Release_Date'] = pd.to_datetime(df['Release_Date'])

                # Extract the date part using .dt.date
                df['Published_date'] = df['Published_date'].dt.date
                df['Release_Date'] = df['Release_Date'].dt.date

                # Filter the DataFrame Before movie release
                df = df[df['Published_date'] <= df['Release_Date'].iloc[0]]

                combined_df = pd.concat([combined_df, df], ignore_index=True)

                # Filter rows with 'Classification' column value equal to "Others"
                filtered_df = df[df['Classification'] == "Others"][['Video_ID', 'Title', 'Views']]

                # Append the filtered DataFrame to the list
                filtered_dfs.append(filtered_df)

                # Check if the classification column exists
                if "Classification" in df.columns:
                    # Iterate over the classification values and their indices
                    for idx, (value, views) in enumerate(zip(df["Classification"], df["Views"]), start=1):
                        # Append the index and views to the classification's order and views list
                        classification_order_views[value].append((idx, views))
                else:
                    st.warning(f"File '{file}' does not contain a 'Classification' column.")
            except Exception as e:
                st.error(f"An error occurred while processing file '{file}': {str(e)}")

    # Convert the defaultdict to a regular dictionary
    classification_order_views_dict = dict(classification_order_views)

    if len(filtered_dfs) >= 1:
        # Combine all filtered DataFrames into a single DataFrame
        combined_filtered_df = pd.concat(filtered_dfs)
    else:
        combined_filtered_df = pd.DataFrame()

    return classification_order_views_dict, combined_filtered_df, combined_df

def get_yt_category(count):
    """
    Returns the view count category based on the given view count.

    Args:
        count (int): The view count.

    Returns:
        str: The view count category.
    """
    try:
        if count >= 1000000:
            return f"{count / 1000000:.1f} Million"
        elif count >= 100000:
            return f"{count / 100000:.1f} Lakh"
        elif count >= 1000:
            return f"{count / 1000:.1f} K"
        else:
            return str(count)
    except Exception as e:
        st.error("An error occurred while determining the view count category:")
        st.exception(e)


def process_promotional_content(classification_order_views_dict):
    try:
        # Sort the classification values based on the minimum index in their order list
        sorted_classification = sorted(classification_order_views_dict,
                                       key=lambda x: min([y[0] for y in classification_order_views_dict[x]]))

        # Calculate the mode value and avg views for each classification
        modes_views = {
            key: (max(Counter([x[0] for x in value]).items(), key=lambda x: x[1])[0],
                  get_view_category(int(sum(x[1] for x in value) / len(value))))
            for key, value in classification_order_views_dict.items()
        }

        # Sort the dictionary based on the first element of the value tuple (order)
        sorted_dict_views = dict(sorted(modes_views.items(), key=lambda kv: kv[1][0]))

        # Create a DataFrame with the content type, its order, and views
        order_of_promotional_content_views = pd.DataFrame(list(sorted_dict_views.items()),
                                                          columns=['Content Type', 'Order_Views'])
        # Create separate 'Order' and 'Views' columns from the 'Order_Views' list
        order_of_promotional_content_views[['Order', 'Views']] = pd.DataFrame(
            order_of_promotional_content_views['Order_Views'].tolist(), index=order_of_promotional_content_views.index)

        # Drop the 'Order_Views' column
        order_of_promotional_content_views.drop(columns='Order_Views', inplace=True)

        st.title("Order of Promotional Content:")
        st.dataframe(order_of_promotional_content_views)
    except Exception as e:
        st.error("An error occurred while determining the process_promotional_content:")
        st.exception(e)


# Assume get_view_category is already defined
def format_timedelta(td):
    days = td.days
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"


def analyze_promotional_content(combined_df):
    try:
        combined_df["Views"].fillna(0, inplace=True)
        combined_df["Previous_Content_Interval"] = pd.to_timedelta(combined_df["Previous_Content_Interval"])
        filtered_df = combined_df[combined_df["Previous_Content_Interval"] >= timedelta(days=1)]

        average_interval = filtered_df["Previous_Content_Interval"].mean()
        max_interval = filtered_df["Previous_Content_Interval"].max()
        min_interval = filtered_df["Previous_Content_Interval"].min()

        st.title("Content Interval Stats")
        st.write("Average Content_Interval:", format_timedelta(average_interval))
        st.write("Maximum Content_Interval:", format_timedelta(max_interval))
        st.write("Minimum Content_Interval:", format_timedelta(min_interval))

        unique_classifications = combined_df["Classification"].unique()
        classification_data = []

        for classification in unique_classifications:
            filtered_classification = combined_df[combined_df["Classification"] == classification]

            if len(filtered_classification) == 0:
                classification_dict = {
                    "Classification": classification,
                    "Max Views": 0,
                    "Min Views": 0,
                    "Avg Views": 0,
                    "Max Likes": 0,
                    "Min Likes": 0,
                    "Avg Likes": 0,
                    "Max Comments": 0,
                    "Min Comments": 0,
                    "Avg Comments": 0
                }
                classification_data.append(classification_dict)
            else:
                max_views = get_view_category(filtered_classification["Views"].max())
                min_views = get_view_category(filtered_classification["Views"].min())
                avg_views = get_view_category(round(filtered_classification["Views"].mean()))

                max_likes = get_view_category(filtered_classification["Likes"].max())
                min_likes = get_view_category(filtered_classification["Likes"].min())
                avg_likes = get_view_category(round(filtered_classification["Likes"].mean()))

                max_comments = get_view_category(filtered_classification["Comments"].max())
                min_comments = get_view_category(filtered_classification["Comments"].min())
                avg_comments = get_view_category(round(filtered_classification["Comments"].mean()))

                classification_dict = {
                    "Classification": classification,
                    "Max Views": max_views,
                    "Min Views": min_views,
                    "Avg Views": avg_views,
                    "Max Likes": max_likes,
                    "Min Likes": min_likes,
                    "Avg Likes": avg_likes,
                    "Max Comments": max_comments,
                    "Min Comments": min_comments,
                    "Avg Comments": avg_comments
                }

                classification_data.append(classification_dict)

        classification_df = pd.DataFrame(classification_data)
        classification_counts = combined_df["Classification"].value_counts(ascending=False)
        classification_df["Counts"] = classification_df["Classification"].map(classification_counts)

        classification_df = classification_df.sort_values(by="Avg Views", ascending=False)
        classification_df = classification_df.reset_index(drop=True)
        classification_df = classification_df[["Classification", "Avg Views", "Avg Likes", "Avg Comments", "Counts"]]

        st.title("Average Views, Likes, and Comments of Each Promotional Content")
        st.dataframe(classification_df)

        classification_counts = combined_df["Classification"].value_counts(ascending=False)

        st.title("Counts of Each Promotional Content")
        # Convert the Series to a DataFrame
        classification_counts_df = classification_counts.reset_index()
        classification_counts_df.columns = ["Classification", "Count"]

        # Display the resulting DataFrame
        st.dataframe(classification_counts_df)
    except Exception as e:
        st.error("An error occurred while determining the analyze_promotional_content:")
        st.exception(e)


def analyze_movie_data(full_subfolder_path, selected_files, classifications_to_check, OG_file_options):
    selected_files = [selected.replace(" ", "") for selected in selected_files]
    processed_results = []

    try:
        # Find files that contain values from lis
        selected_file_path = [file for file in OG_file_options if any(value in file for value in selected_files)]

        for matching_file in selected_file_path:
            file_path = os.path.join(full_subfolder_path, matching_file)

            df = pd.read_excel(file_path)

            # Filter rows with specified classifications
            filtered_rows = df[df['Classification'].isin(classifications_to_check)]

            if not filtered_rows.empty:
                # Get the release date from the first row
                release_date = pd.to_datetime(df['Release_Date'].iloc[0])

                for vid in filtered_rows['Video_ID'].unique():
                    sub_data = filtered_rows[filtered_rows['Video_ID'] == vid]

                    published_date = pd.to_datetime(sub_data['Published_date'].iloc[0])
                    classification = sub_data['Classification'].iloc[0]
                    views = sub_data['Views'].iloc[0]

                    # Determine if entry is 'Before' or 'After' release date
                    status = 'Before' if published_date < release_date else 'After'

                    # Calculate the difference in days between dates
                    date_difference = (published_date - release_date).days

                    modified_movie_name = matching_file.replace("_", " ").replace(".xlsx", "").replace(".xls", "")

                    # Append processed result to the list
                    processed_results.append({
                        'Movie': modified_movie_name,
                        'Video_ID': vid,
                        'Classification': classification,
                        'Published_Date': published_date,
                        'Release_Date': release_date,
                        'Status': status,
                        'Views': views,
                        'Date_Difference': date_difference
                    })

        # Create a DataFrame from the processed results list
        result_df = pd.DataFrame(processed_results)

        # Clean up 'Movie' column
        result_df['Movie'] = result_df['Movie'].astype(str).str.replace(r'_', '', regex=True).str.replace('.xlsx', '',
                                                                                                          regex=True)

        # Group by movie, classification, and status, and aggregate data
        result_df = result_df.groupby(['Movie', 'Video_ID', 'Classification', 'Status']).agg({
            'Date_Difference': 'mean',
            'Views': 'max'
        }).reset_index()

        # Apply get_view_category function to 'Views' column
        result_df['Views'] = result_df['Views'].apply(get_view_category)

        return result_df

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
        return pd.DataFrame()

def analyze_all_movie_data(full_subfolder_path, files, classifications_to_check):
    processed_results = []

    try:
        # Iterate over each file
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                # Construct the full file path
                file_path = os.path.join(full_subfolder_path, file)

                # Read the Excel file into a pandas DataFrame
                df = pd.read_excel(file_path)

                # Filter rows with specified classifications
                filtered_rows = df[df['Classification'].isin(classifications_to_check)]

                if not filtered_rows.empty:
                    # Get the release date from the first row
                    release_date = pd.to_datetime(df['Release_Date'].iloc[0])

                    for vid in filtered_rows['Video_ID'].unique():
                        sub_data = filtered_rows[filtered_rows['Video_ID'] == vid]

                        published_date = pd.to_datetime(sub_data['Published_date'].iloc[0])
                        classification = sub_data['Classification'].iloc[0]
                        views = sub_data['Views'].iloc[0]

                        # Determine if entry is 'Before' or 'After' release date
                        status = 'Before' if published_date < release_date else 'After'

                        # Calculate the difference in days between dates
                        date_difference = (published_date - release_date).days

                        modified_movie_name = file.replace("_", " ").replace(".xlsx", "").replace(".xls", "")

                        # Append processed result to the list
                        processed_results.append({
                            'Movie': modified_movie_name,
                            'Video_ID': vid,
                            'Classification': classification,
                            'Published_Date': published_date,
                            'Release_Date': release_date,
                            'Status': status,
                            'Views': views,
                            'Date_Difference': date_difference
                        })

        # Create a DataFrame from the processed results list
        result_df = pd.DataFrame(processed_results)

        # Clean up 'Movie' column
        result_df['Movie'] = result_df['Movie'].str.replace(r'_', '', regex=True).str.replace('.xlsx', '', regex=True)

        # Group by movie, classification, and status, and aggregate data
        result_df = result_df.groupby(['Movie', 'Video_ID', 'Classification', 'Status']).agg({
            'Date_Difference': 'mean',
            'Views': 'max'
        }).reset_index()

        # Apply get_view_category function to 'Views' column
        result_df['Views'] = result_df['Views'].apply(get_view_category)

        return result_df

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()

def get_order_of_promotional_content(full_subfolder_path, files, selected_files, OG_file_options):
    # Dictionary to store the order of appearance and views for each classification value
    classification_order_views = defaultdict(list)

    combined_df = pd.DataFrame()

    # List to store filtered DataFrames
    filtered_dfs = []

    selected_files = [selected.replace(" ", "") for selected in selected_files]

    # Find files that contain values from lis
    selected_file_path = [file for file in OG_file_options if any(value in file for value in selected_files)]

    for matching_file in selected_file_path:
        file_path = os.path.join(full_subfolder_path, matching_file) #os.path.join(directory, matching_file)
        data = pd.read_excel(file_path)

        data['Published_date'] = pd.to_datetime(data['Published_date'])
        data['Release_Date'] = pd.to_datetime(data['Release_Date'])

        data['Published_date'] = pd.to_datetime(data['Published_date'])
        data['Release_Date'] = pd.to_datetime(data['Release_Date'])

        # Extract the date part using .dt.date
        data['Published_date'] = data['Published_date'].dt.date
        data['Release_Date'] = data['Release_Date'].dt.date
        # Filter the DataFrame Before movie release
        data = data[data['Published_date'] <= data['Release_Date'].iloc[0]]

        combined_df = pd.concat([combined_df, data], ignore_index=True)

        filtered_df = data[data['Classification'] == "Others"][['Video_ID', 'Title', 'Views']]

        # Append the filtered DataFrame to the list
        filtered_dfs.append(filtered_df)

        # Check if the classification column exists
        if "Classification" in data.columns:
            # Iterate over the classification values and their indices
            for idx, (value, views) in enumerate(zip(data["Classification"], data["Views"]), start=1):
                # Check if there's a previous entry for the same classification value and index
                existing_entry = next(((existing_idx, existing_views) for existing_idx, existing_views in classification_order_views[value] if existing_idx == idx), None)

                if existing_entry is not None:
                    # If there's an existing entry, update the views
                    existing_idx, existing_views = existing_entry
                    updated_views = existing_views + views
                    classification_order_views[value].remove((existing_idx, existing_views))  # Remove the old entry
                    classification_order_views[value].append((existing_idx, updated_views))    # Add the updated entry
                else:
                    # If there's no existing entry, add a new entry
                    classification_order_views[value].append((idx, views))
        else:
            st.warning(f"File '{selected_file_path}' does not contain a 'Classification' column.")

    # Convert the defaultdict to a regular dictionary
    classification_order_views_dict = dict(classification_order_views)

    if len(filtered_dfs) >= 1:
        # Combine all filtered DataFrames into a single DataFrame
        combined_filtered_df = pd.concat(filtered_dfs)
    else:
        combined_filtered_df = pd.DataFrame()

    return classification_order_views_dict, combined_filtered_df, combined_df

# Main Streamlit app
def main():

    st.title("Movie Promotion Analysis App")

    # Select main directory
    directory_paths = [r"C:\Users\mohan\Desktop\EMR\Bhai Movie Promotional Content Planning\Related_Movie_Content_Data"]
    # choice_main_directory = st.sidebar.selectbox("Select Movie:", [os.path.basename(path) for path in directory_paths])
    #
    # main_directory_path = choice_main_directory #directory_paths[0]  # Assuming only one main directory for simplicity
    # main_directory_name = os.path.basename(main_directory_path)

    #st.sidebar.write(f"You've selected: {main_directory_name}")

    choice_main_directory = st.sidebar.selectbox("Select Movie:",
                                                 [None] + [os.path.basename(pat) for path in directory_paths for pat in
                                                           os.listdir(path)])

    if choice_main_directory is None:
        st.sidebar.write("Please select a movie.")
    else:
        main_directory_path = os.path.join(directory_paths[0], choice_main_directory)
        main_directory_name = os.path.basename(main_directory_path)
        st.sidebar.write(f"You've selected: **{main_directory_name}**")

        if main_directory_name.lower() == 'bhai':
            st.title(main_directory_name.title())
            video_url = "https://www.youtube.com/embed/bhjChcKBGwI?autoplay=1"

            # Use Markdown with inline CSS to adjust the position
            st.markdown(
                f'<iframe width="740" height="460" src="{video_url}" frameborder="1" allowfullscreen autoplay></iframe>',
                unsafe_allow_html=True)


        # List subdirectories in the chosen main directory
        subdirectories = [d for d in os.listdir(main_directory_path) if os.path.isdir(os.path.join(main_directory_path, d))]

        # Select subdirectory
        choice_subfolder = st.sidebar.selectbox("Select Region:", [None] + subdirectories)

        if choice_subfolder is None:
            st.sidebar.write("Please select a region.")
        else:
            selected_subfolder = choice_subfolder
            full_subfolder_path = os.path.join(main_directory_path, selected_subfolder)
            st.sidebar.write(f"You've selected the Region: **{selected_subfolder}**")

            # Get a list of all files in the subdirectory
            files = os.listdir(full_subfolder_path)

            # Display available files to the user
            st.sidebar.write("Available Files:")
            file_options = [file.replace("_", " ").replace(".xlsx", "").replace(".xls", "") for file in files]
            file_options.append("Overall")  # Add 'Overall' option to the list
            selected_files = st.sidebar.multiselect("Select Files:", file_options)

            selected_files = [selected.replace(" ", "") for selected in selected_files]
            OG_file_options = [file for file in files if file.endswith(('.xlsx', '.xls'))]

            # Process selected files
            if st.sidebar.button("Process Selected Files"):
                with st.spinner("Processing..."):
                    if len(selected_files) == 1 and "Overall" in selected_files:
                        # Process all files for "Overall" option
                        combined_df = pd.DataFrame()

                        st.title("\nMovie Engagement:")
                        for file in files:
                            if file.endswith(".xlsx") or file.endswith(".xls"):
                                file_path = os.path.join(full_subfolder_path, file)
                                df = pd.read_excel(file_path)
                                modified_movie_name = file.replace("_", "").replace(".xlsx", "").replace(".xls", "")

                                calculate_cumulative_statistics(df, modified_movie_name)

                        classification_order_views_dict, combined_filtered_df, combined_df = get_order_of_promotional_content_overall(
                            full_subfolder_path, files)

                        classifications_to_check = ['Video Song', 'Movie Video Song']
                        result_df = analyze_all_movie_data(full_subfolder_path, files, classifications_to_check)

                    else:
                        combined_df = pd.DataFrame()
                        st.title("\nMovie Engagement:")

                        # Remove the item "jawan" from the list
                        if "Overall" in selected_files:
                            selected_files.remove("Overall")

                        # Find files that contain values from lis
                        selected_file_path = [file for file in OG_file_options if any(value in file for value in selected_files)]

                        for matching_file in selected_file_path:
                            file_path = os.path.join(full_subfolder_path, matching_file)

                            df = pd.read_excel(file_path)

                            modified_movie_name = matching_file.replace("_", "").replace(".xlsx", "").replace(".xls", "")

                            calculate_cumulative_statistics(df, modified_movie_name)

                        classification_order_views_dict, combined_filtered_df, combined_df = get_order_of_promotional_content(
                            full_subfolder_path, files, selected_files, OG_file_options)

                        classifications_to_check = ['Video Song', 'Movie Video Song']
                        result_df = analyze_movie_data(full_subfolder_path, selected_files, classifications_to_check, OG_file_options)

                    process_promotional_content(classification_order_views_dict)

                    st.title("\nOthers Classified Content:")
                    # Display the sorted and indexed DataFrame
                    if 'combined_filtered_df' in locals() and len(combined_filtered_df) >= 1:
                        # Sort the combined DataFrame in descending order based on 'Views' column
                        combined_filtered_df = combined_filtered_df.sort_values(by='Views', ascending=False)
                        # Reset the index of the DataFrame
                        combined_filtered_df = combined_filtered_df.reset_index(drop=True)

                        st.dataframe(combined_filtered_df)

                    analyze_promotional_content(combined_df)

                    if 'result_df' in locals() and len(result_df) >= 1:
                        st.title("\nSongs Release Stat:")
                        st.dataframe(result_df)


if __name__ == "__main__":
    main()