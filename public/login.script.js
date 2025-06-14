document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    const passwordInput = document.getElementById('password');
    const loginButton = document.getElementById('login-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMessageDiv = document.getElementById('error-message');
    const errorTextSpan = document.getElementById('error-text');
    
    function showLoading() {
        loginButton.disabled = true;
        loadingSpinner.classList.remove('hidden');
    }
    
    function hideLoading() {
        loginButton.disabled = false;
        loadingSpinner.classList.add('hidden');
    }
    
    function showError(message) {
        errorTextSpan.textContent = message;
        errorMessageDiv.classList.remove('hidden');
    }
    
    function hideError() {
        errorMessageDiv.classList.add('hidden');
        errorTextSpan.textContent = '';
    }
    
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        hideError();
        showLoading();
        const password = passwordInput.value;
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ password }),
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                localStorage.setItem('isLoggedIn', 'true');
                window.location.href = '/admin';
            } else {
                showError(data.error || 'Login failed. Please check your password.');
                passwordInput.focus();
            }
        } catch (error) {
            console.error('Login Error:', error);
            showError('An error occurred during login. Please try again.');
        } finally {
            hideLoading();
        }
    });
});