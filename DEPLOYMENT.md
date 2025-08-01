# Render Deployment Guide for DocLore

## 🚀 Quick Deploy to Render

### 1. **Prerequisites**
- Render account
- GitHub repository with your code
- Environment variables ready

### 2. **Environment Variables to Set in Render**
```bash
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
RENDER=true
```

### 3. **Deployment Steps**

1. **Connect your GitHub repo to Render**
2. **Create a new Web Service**
3. **Use the render.yaml configuration** (already included)
4. **Set environment variables** in Render dashboard
5. **Deploy!**

## 🔧 **Render Optimizations Implemented**

### **File Download Enhancements**
- ✅ **Large file support**: Up to 100MB (50MB on Render)
- ✅ **Streaming downloads**: Memory-efficient chunked downloads
- ✅ **Timeout handling**: 120-second timeout for large files
- ✅ **Size validation**: Pre-download size checks

### **Memory Management**
- ✅ **Batch processing**: Chunks processed in small batches
- ✅ **Memory limits**: Configurable file size limits
- ✅ **Cache optimization**: Embedding cache with disk storage
- ✅ **Render detection**: Automatic optimization when on Render

### **Performance Optimizations**
- ✅ **Local embeddings**: No API quotas, faster processing
- ✅ **Caching system**: Reduces redundant computations
- ✅ **Health checks**: Render-compatible health endpoint
- ✅ **Error handling**: Comprehensive error management

## 📊 **Resource Usage**

### **Memory Limits**
- **Local development**: 100MB file limit
- **Render deployment**: 50MB file limit
- **Chunk size**: 4KB on Render, 8KB locally

### **Processing Limits**
- **Timeout**: 120 seconds for large files
- **Batch size**: 10 chunks on Render, 20 locally
- **Cache storage**: 1GB disk space allocated

## 🛠 **Troubleshooting**

### **Common Issues**

1. **File too large error**
   - Solution: Reduce file size or use smaller documents
   - Limit: 50MB on Render, 100MB locally

2. **Timeout errors**
   - Solution: Check network connectivity
   - Increase timeout in render.yaml if needed

3. **Memory errors**
   - Solution: Reduce batch size in main.py
   - Monitor memory usage in Render dashboard

### **Monitoring**
- Health check: `https://your-app.onrender.com/health`
- Logs: Available in Render dashboard
- Metrics: CPU/Memory usage in dashboard

## 🔄 **Updates and Maintenance**

### **Updating the System**
1. Push changes to GitHub
2. Render auto-deploys on main branch
3. Monitor deployment logs
4. Test health endpoint

### **Scaling**
- Current: Starter plan (512MB RAM)
- Upgrade: Pro plan for more resources
- Custom: Enterprise for high-volume usage

## 📝 **API Usage**

### **Endpoint**
```
POST /api/v1/hackrx/run
```

### **Headers**
```
Authorization: Bearer 10fbd8807c6d9b5a37028c3eb4bd885959f06d006aedd2dc5ba64c5ccea913c0
Content-Type: application/json
```

### **Request Body**
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": ["What are the key terms?", "What is the coverage amount?"]
}
```

## 🎯 **Performance Tips**

1. **Use smaller documents** for faster processing
2. **Batch questions** instead of multiple requests
3. **Monitor cache usage** in embedding_cache directory
4. **Check Render logs** for performance insights

## 🔒 **Security**

- ✅ **Authentication**: Bearer token required
- ✅ **File validation**: Type and size checks
- ✅ **Error sanitization**: No sensitive data in errors
- ✅ **Rate limiting**: Consider adding if needed

---

**Ready to deploy!** 🚀

Your system is now optimized for Render with large file support, memory management, and performance optimizations. 