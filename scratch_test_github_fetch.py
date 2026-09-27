from app.github.repo_reader import fetch_file_content

content = fetch_file_content(
    owner="deepalsr",
    repo="ai_rag_assistant",
    path="README.md",
    ref="main",
)

print("CONTENT:")
print(content)