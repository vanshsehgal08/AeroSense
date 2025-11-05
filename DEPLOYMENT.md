# Deployment Guide for AeroSense on Render

This guide will walk you through deploying AeroSense on Render.

## Prerequisites

1. A GitHub account
2. Your project pushed to a GitHub repository
3. A Render account (sign up at [render.com](https://render.com))

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure all files are committed and pushed to GitHub:
- `Procfile` - Web server configuration
- `render.yaml` - Render deployment configuration
- `requirements.txt` - Python dependencies (includes gunicorn)
- `build.sh` - Build script (optional, can use render.yaml)
- All project files

### 2. Create Render Account & New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if not already connected
4. Select your repository containing AeroSense

### 3. Configure Service Settings

**Basic Settings:**
- **Name**: `aerosense` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)

**Build & Deploy:**
- **Build Command**: 
  ```bash
  pip install -r requirements.txt && python -m spacy download en_core_web_sm && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')" && cd frontend && npm install && npm run build && cd ..
  ```
- **Start Command**: 
  ```bash
  gunicorn app:app
  ```

**Environment Variables:**
Add these environment variables in Render dashboard:
- `PYTHON_VERSION`: `3.11.0`
- `FLASK_ENV`: `production`
- `REACT_APP_API_URL`: Leave empty (will use relative URLs in production)

**Advanced Settings:**
- **Health Check Path**: `/api/health`
- **Auto-Deploy**: `Yes` (deploys on every push)

### 4. Alternative: Using render.yaml (Recommended)

If you prefer automated configuration:

1. Make sure `render.yaml` is in your repository root
2. In Render dashboard, select **"Apply render.yaml"**
3. Render will automatically detect and use the configuration

### 5. Deploy

1. Click **"Create Web Service"**
2. Render will start building your application
3. The build process includes:
   - Installing Python dependencies
   - Downloading spaCy model
   - Downloading NLTK data
   - Installing Node.js dependencies
   - Building React frontend
4. Monitor the build logs for any errors

### 6. Update Frontend API Configuration

After deployment, update the API URL in your frontend:

1. Go to Render dashboard → Your service → Environment
2. Add environment variable:
   - Key: `REACT_APP_API_URL`
   - Value: `https://your-app-name.onrender.com/api` (replace with your actual URL)

3. **OR** update `frontend/src/services/api.js` to use relative URLs:
   ```javascript
   const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';
   ```

### 7. Post-Deployment Setup

1. **First Training**: After deployment, you'll need to train the model:
   - Upload a CSV file with reviews
   - Go to "Train Model" tab
   - Train the model with your data

2. **Verify Deployment**:
   - Visit your Render URL
   - Check `/api/health` endpoint
   - Test prediction functionality

## Configuration Files

### Procfile
```
web: gunicorn app:app
```

### render.yaml
Contains all deployment configuration. See file for details.

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | `3.11.0` | Python version |
| `FLASK_ENV` | `production` | Flask environment |
| `PORT` | Auto-set by Render | Server port |
| `REACT_APP_API_URL` | (optional) | API URL for frontend |

## Troubleshooting

### Build Fails

1. **spaCy model download fails**:
   - Check build logs
   - Ensure `python -m spacy download en_core_web_sm` is in build command

2. **NLTK data missing**:
   - Ensure NLTK downloads are in build command
   - Check that `nltk.download()` is called in app.py

3. **Frontend build fails**:
   - Check Node.js version (Render uses Node 18+ by default)
   - Verify `package.json` is correct
   - Check for missing dependencies

### Runtime Errors

1. **Model not found**:
   - Train the model after deployment
   - Upload data and train via UI

2. **CORS errors**:
   - Already handled by `flask-cors`
   - If issues persist, check CORS configuration in app.py

3. **Database errors**:
   - SQLite database is created automatically
   - Ensure write permissions (Render handles this)

### Performance

1. **Slow first request**:
   - Normal due to cold starts on free tier
   - Consider upgrading to paid plan for better performance

2. **Build timeout**:
   - Free tier has 20-minute build limit
   - Optimize build by caching dependencies
   - Consider splitting build into stages

## Free Tier Limitations

- **Cold starts**: Free services sleep after 15 minutes of inactivity
- **Build time**: 20-minute limit
- **Memory**: 512MB RAM limit
- **CPU**: Shared resources

**Recommendation**: Upgrade to paid plan for production use.

## Custom Domain (Optional)

1. Go to your service → Settings → Custom Domain
2. Add your domain
3. Update DNS records as instructed
4. SSL is automatically provisioned

## Monitoring

- View logs in Render dashboard
- Set up alerts for service failures
- Monitor health check endpoint

## Continuous Deployment

Render automatically deploys on every push to your main branch when "Auto-Deploy" is enabled.

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
- Check build logs for specific errors

---

**Note**: The first deployment may take 10-15 minutes due to dependency installation and model downloads. Subsequent deployments are faster.

