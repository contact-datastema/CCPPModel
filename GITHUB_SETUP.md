# GitHub Setup Instructions

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and log in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the details:
   - **Repository name:** `CCPPModel`
   - **Description:** "Machine Learning model to predict Combined Cycle Power Plant energy output"
   - **Visibility:** Public (or Private, your choice)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 2: Link Local Repository to GitHub

After creating the repository on GitHub, run these commands in your terminal:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/CCPPModel.git

# Verify the remote was added
git remote -v

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Verify

Go to `https://github.com/YOUR_USERNAME/CCPPModel` to see your repository!

## What's Included in the Repository

✅ Source code (`src/` directory)
✅ Data file (`data/CCPP_data.csv`)
✅ Requirements file (`requirements.txt`)
✅ README with instructions
✅ .gitignore (excludes venv, plots, etc.)

## What's Excluded (in .gitignore)

❌ Virtual environments (`venv/`, `.venv/`)
❌ Generated plots (`plots/`)
❌ Presentation materials (`presentation_plan.md`, `index.html`)
❌ Python cache files (`__pycache__/`, `*.pyc`)

## Alternative: Using GitHub CLI

If you have GitHub CLI installed, you can create the repository directly from terminal:

```bash
gh repo create CCPPModel --public --source=. --remote=origin --push
```

This will create the repository and push your code in one command!
