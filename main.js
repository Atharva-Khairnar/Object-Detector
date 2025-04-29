document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const uploadZone = document.getElementById('upload-zone');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const uploadBtn = document.getElementById('upload-btn');
    const loadingContainer = document.getElementById('loading-container');
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        if (fileInput.files && fileInput.files[0]) {
            // Show preview of selected image
            const reader = new FileReader();
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                uploadBtn.style.display = 'block';
            };
            
            reader.readAsDataURL(fileInput.files[0]);
        }
    });
    
    // Handle drag and drop
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.classList.add('highlight');
    });
    
    uploadZone.addEventListener('dragleave', function() {
        uploadZone.classList.remove('highlight');
    });
    
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('highlight');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            fileInput.files = e.dataTransfer.files;
            
            // Trigger change event
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });
    
    // Click on upload zone to trigger file input
    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Show loading spinner when form is submitted
    uploadForm.addEventListener('submit', function() {
        // Validate file is selected
        if (!fileInput.files || !fileInput.files[0]) {
            alert('Please select an image file first.');
            return false;
        }
        
        // Validate file type
        const fileType = fileInput.files[0].type;
        if (!fileType.match('image/jpeg') && !fileType.match('image/png')) {
            alert('Please upload a JPEG or PNG image.');
            return false;
        }
        
        // Validate file size (max 16MB)
        if (fileInput.files[0].size > 16 * 1024 * 1024) {
            alert('File size exceeds 16MB. Please upload a smaller image.');
            return false;
        }
        
        // Show loading spinner and hide upload button
        loadingContainer.style.display = 'block';
        uploadBtn.style.display = 'none';
        uploadZone.style.display = 'none';
    });
    
    // Handle alerts automatic dismissal
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});
