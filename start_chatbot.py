import os
import sys

def check_requirements():
    required_files = [
        "data/knowledge_base.json",
        "data/processed_training_data.json", 
        "terminal_chatbot.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")

        if os.path.exists("data/knowledge_base.json"):
            print("\n💡 Basic version can work with just knowledge base")
            return "basic"
        else:
            print("\n❌ Cannot start. Please ensure required files exist.")
            return False

    return True

def check_dependencies():
    required_packages = [
        ("nltk", "Natural Language Toolkit"),
        ("sklearn", "Machine Learning Library"), 
        ("numpy", "Numerical Computing"),
        ("fuzzywuzzy", "String Matching")
    ]

    missing_packages = []

    for package, description in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append((package, description))

    if missing_packages:
        print("❌ Missing required Python packages:")
        for package, desc in missing_packages:
            print(f"   - {package} ({desc})")

        print("\n💡 Install missing packages with:")
        print("pip install nltk scikit-learn numpy fuzzywuzzy")
        return False

    return True

def main():
    print("🌾 Agricultural Chatbot Startup")
    print("=" * 40)

    print("🔍 Checking Python dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        print("Command: pip install nltk scikit-learn numpy fuzzywuzzy")
        return False

    print("✅ All dependencies found!")

    print("\n📋 Checking required files...")
    file_check = check_requirements()

    if file_check == False:
        return False
    elif file_check == "basic":
        print("⚠️  Running in basic mode (without enhanced training data)")
    else:
        print("✅ All files found!")

    print("\n🚀 Starting Agricultural Chatbot...")
    print("-" * 40)

    try:
        from terminal_chatbot import main as run_chatbot
        run_chatbot()
    except ImportError as e:
        print(f"❌ Error importing chatbot: {e}")
        print("Please ensure terminal_chatbot.py is in the current directory.")
        return False
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        return False

if __name__ == "__main__":
    main()
