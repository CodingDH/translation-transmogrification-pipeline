# Error Logs Datasets

This repository contains all error logs for our datasets. Depending on the error the file will be formatted slightly differently but we have four categories of error logs:

## Search Files

This folder contains either errors from our search queries or if the code stops we also write where the code stopped, so we don't have to rerun all search queries.

## Join Files

This folder contains errors from our join queries.

## Entity Files

This folder contains errors from our entity queries.

## Entity Metadata Files

This folder contains errors from our entity metadata queries.

**DO NOT PUSH UP UNLESS RE-RUNNING FULL DATA PIPELINE**

## Data Folder Structure

The code assumes the following folder structure. The asterisks indicate this repository's folders.:

```Markdown
├── searched_issue_data/
│   ├── public_history/
│   ├── cultural_analytics/
│   ├── computational_humanities/
│   ├── computational_social_science/
│   ├── digital_history/
│   ├── digital_cultural_heritage/
│   └── digital_humanities/
├── collected_datasets/
├── derived_files/
├── searched_repo_data/
│   ├── humanities/
│   ├── public_history/
│   ├── cultural_analytics/
│   ├── computational_humanities/
│   ├── computational_social_science/
│   ├── digital_history/
│   ├── digital_cultural_heritage/
│   ├── digital_humanities/
├── **error_logs/**
│   ├── entity_metadata_files/
│   ├── search_files/
│   ├── join_files/
│   │   └── older_files/
│   └── entity_files/
├── join_files/
├── entity_files/
├── threshold_reached_results/
│   ├── automated_thresholding/
│   └── manual_thresholding/
├── searched_user_data/
│   ├── humanities/
│   ├── public_history/
│   ├── cultural_analytics/
│   ├── computational_humanities/
│   ├── computational_social_science/
│   ├── digital_history/
│   ├── digital_cultural_heritage/
│   └── digital_humanities/
├── metadata_files/
│   ├── older_terms_json/
│   ├── github_api_examples/
│   │   └── version_2022_11_28/
│   ├── translated_terms/
│   │   ├── humanities_data_science/
│   │   ├── humanities/
│   │   ├── public_history/
│   │   ├── cultural_analytics/
│   │   ├── computational_humanities/
│   │   ├── computational_social_science/
│   │   ├── digital_history/
│   │   ├── digital_cultural_heritage/
│   │   └── digital_humanities/
├── historic_data/
│   ├── repo_metadata/
│   │   ├── repo_languages/
│   │   ├── repo_profile/
│   │   └── repo_commits_join_dataset/
│   ├── join_files/
│   │   ├── user_following_join_dataset/
│   │   ├── repo_forks_join_dataset/
│   │   ├── repo_pulls_join_dataset/
│   │   ├── org_repos_join_dataset/
│   │   ├── org_followers_join_dataset/
│   │   ├── repo_subscribers_join_dataset/
│   │   ├── user_repos_join_dataset/
│   │   ├── user_followers_join_dataset/
│   │   ├── org_members_join_dataset/
│   │   ├── repo_issues_join_dataset/
│   │   ├── user_subscriptions_join_dataset/
│   │   ├── user_gists_join_dataset/
│   │   ├── repo_stargazers_join_dataset/
│   │   ├── issue_comments_join_dataset/
│   │   ├── user_starred_join_dataset/
│   │   ├── user_starred_join_dataset/
│   │   ├── repo_orgs_join_dataset/
│   │   ├── pull_comments_join_dataset/
│   │   ├── repo_comments_join_dataset/
│   │   ├── repo_contributors_join_dataset/
│   │   └── user_orgs_join_dataset/
│   ├── entity_files/
│   │   ├── all_gists/
│   │   │   └── collected_datasets/
│   │   ├── all_users/
│   │   │   └── collected_datasets/
│   │   ├── all_orgs/
│   │   │   └── collected_datasets/
│   │   ├── all_repos/
│   │   │   └── collected_datasets/
│   │   │   ├── collected_datasets/
```