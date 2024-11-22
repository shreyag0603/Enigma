$(document).ready(function() {
    // Initialize carousel
    $('#videoCarousel').carousel({
      interval: false, // Disable automatic cycling
      ride: 'carousel'
    });

    // Play video on slide change
    $('#videoCarousel').on('slide.bs.carousel', function(e) {
      var video = $(e.relatedTarget).find('video')[0];
      if (video) {
        video.play(); // Autoplay the video on slide change
      }
    });

    // When the video ends, move to the next slide
    $('#videoCarousel video').on('ended', function() {
      $('#videoCarousel').carousel('next');
    });

    // Play video on slide change and handle previous video
    $('#videoCarousel').on('slide.bs.carousel', function(e) {
      var slideFrom = $(this).find('.active').index();
      var slideTo = $(e.relatedTarget).index();
      var video = $(e.relatedTarget).find('video')[0];
      if (video) {
        video.play(); // Autoplay the video on slide change
      }
      if (slideFrom !== slideTo) {
        var prevVideo = $(this).find('.carousel-item').eq(slideFrom).find('video')[0];
        if (prevVideo) {
          prevVideo.pause(); // Pause the previous video
          prevVideo.currentTime = 0; // Reset video to start
        }
      }
    });
  });