import uvicorn

if __name__ == "__main__":
    print("\nOpen this URL in your browser (http, not https):")
    print("  http://127.0.0.1:8000/\n")
    # reload_dirs restricts the file-watcher to actual app source code.
    # Without this it recursively watches the whole backend/ folder,
    # including venv/ (thousands of installed-package files) -- inside a
    # OneDrive-synced folder, OneDrive's background sync touches those files
    # constantly, which was triggering nonstop spurious reload storms.
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["app"])
