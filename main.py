#!/usr/bin/env python3
"""
UIUC CourseBot - Main Entry Point
Interactive command-line interface for the course assistant.
"""

from chatbot import UIUCChatBot

def main():
    """Main entry point for command-line interface"""
    try:
        bot = UIUCChatBot()
        bot.run_interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()