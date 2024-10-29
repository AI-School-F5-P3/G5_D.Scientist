document.addEventListener('DOMContentLoaded', function() {
    const carousel = document.getElementById('image-carousel');
    const images = carousel.getElementsByClassName('carousel-image');
    let currentIndex = 0;

    function showNextImage() {
        images[currentIndex].classList.remove('active');
        currentIndex = (currentIndex + 1) % images.length;
        images[currentIndex].classList.add('active');
    }

    // Mostrar la primera imagen
    images[0].classList.add('active');

    // Cambiar de imagen cada 5 segundos
    setInterval(showNextImage, 5000);
});