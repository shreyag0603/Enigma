// subscription.js
document.addEventListener('DOMContentLoaded', function() {
    const proButton = document.getElementById('proButton');
    const businessButton = document.getElementById('businessButton');

    function handleRedirect(planType) {
        // Redirect to the appropriate page based on subscription plan
        if (planType === 'pro') {
            window.location.href = '/transactionpro.html';
        } else if (planType === 'premium') {
            window.location.href = '/transactionpremium.html';
        }
    }

    function handleModalDisplay() {
        const modal = document.getElementById('selectHowToProceedModal');
        if (modal) {
            $('#selectHowToProceedModal').modal('show');
        }
    }

    // Add event listeners to the buttons
    if (proButton) {
        proButton.addEventListener('click', function() {
            handleRedirect('pro');
        });
    }

    if (businessButton) {
        businessButton.addEventListener('click', function() {
            handleRedirect('premium');
        });
    }

    // Check if user has completed payment and show modal
    const isPaymentCompleted = sessionStorage.getItem('paymentCompleted');
    if (isPaymentCompleted === 'true') {
        handleModalDisplay();
        sessionStorage.removeItem('paymentCompleted');
    }
});
