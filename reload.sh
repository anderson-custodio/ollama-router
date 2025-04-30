launchctl unload ~/Library/LaunchAgents/com.anderson.custodio.start_ollama_router.plist
launchctl load ~/Library/LaunchAgents/com.anderson.custodio.start_ollama_router.plist
tail -f /tmp/ollama_router.log
