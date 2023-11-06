
// $(document).ready(function() {
//     $(".form-ajax").on('submit', function (e) {
//         var form = $(this)
//         e.preventDefault();
//         var url = form.attr('action');
//         var data = form.serialize();
//         $.post(url, data, function (result) {
//             if(form.find('.form-results-text').length > 0){
//                 form.find('.form-results-text').text(result);
//                 form.find('.form-results-text').show();
//             }
//             else if(form.find('.form-results-img').length > 0){
//                 pathArray = result.split(',');
//                 imgArray = form.find('.form-results-img img');
//                 for (let i = 0; i < imgArray.length; i++) {
//                     var imgurl = pathArray[i] + '?' + new Date().getTime(); // force img refresh
//                     $(imgArray[i]).attr("src", imgurl);
//                 } 
//                 form.find('.form-results-img').show();
//             }
//         });
//     });
// });


$(document).ready(function() {
    $(".form-ajax").on('submit', function (e) {
        var form = $(this);
        e.preventDefault();
        var url = form.attr('action');
        var data = form.serialize();
        $.post(url, data, function (result) {
            if (form.find('.form-results-text').length > 0 && form.find('.form-results-img').length > 0) {
                // Handle both text and image results
                var results = result.split(';');
                form.find('.form-results-text').text(results[0]);
                form.find('.form-results-text').show();
                
                var pathArray = results[1].split(',');
                var imgArray = form.find('.form-results-img img');
                for (let i = 0; i < imgArray.length; i++) {
                    var imgurl = pathArray[i] + '?' + new Date().getTime(); // Force image refresh
                    $(imgArray[i]).attr("src", imgurl);
                }
                form.find('.form-results-img').show();
            } else if (form.find('.form-results-text').length > 0) {
                // Handle text result only
                form.find('.form-results-text').text(result);
                form.find('.form-results-text').show();
                form.find('.form-results-img').hide();
            } else if (form.find('.form-results-img').length > 0) {
                // Handle image result only
                var pathArray = result.split(',');
                var imgArray = form.find('.form-results-img img');
                for (let i = 0; i < imgArray.length; i++) {
                    var imgurl = pathArray[i] + '?' + new Date().getTime(); // Force image refresh
                    $(imgArray[i]).attr("src", imgurl);
                }
                form.find('.form-results-img').show();
            }
        });
    });
});

//  debug mode considered

// $(document).ready(function() {
//     $(".form-ajax").on('submit', function (e) {
//         var form = $(this);
//         e.preventDefault();
//         var url = form.attr('action');
//         var data = form.serialize();
//         $.post(url, data, function (result) {
//             if (form.find('.form-results-text').length > 0 && form.find('.form-results-img').length > 0) {
//                 // Handle both text and image results
//                 var results = result.split(';');
//                 form.find('.form-results-text').text(results[0]);
//                 form.find('.form-results-text').show();
                
//                 var pathArray = results[1].split(',');
//                 var imgArray = form.find('.form-results-img img');
//                 var numImages = pathArray.length;
//                 if (results[0].includes("debugmode")) {
//                     numImages += 2; // Add 2 more images for debug mode
//                 }
//                 for (let i = 0; i < numImages; i++) {
//                     var imgurl = pathArray[i % pathArray.length] + '?' + new Date().getTime(); // Force image refresh
//                     $(imgArray[i]).attr("src", imgurl);
//                 }
//                 form.find('.form-results-img').show();
//             } else if (form.find('.form-results-text').length > 0) {
//                 // Handle text result only
//                 form.find('.form-results-text').text(result);
//                 form.find('.form-results-text').show();
//                 form.find('.form-results-img').hide();
//             } else if (form.find('.form-results-img').length > 0) {
//                 // Handle image result only
//                 var pathArray = result.split(',');
//                 var imgArray = form.find('.form-results-img img');
//                 for (let i = 0; i < imgArray.length; i++) {
//                     var imgurl = pathArray[i % pathArray.length] + '?' + new Date().getTime(); // Force image refresh
//                     $(imgArray[i]).attr("src", imgurl);
//                 }
//                 form.find('.form-results-img').show();
//             }
//         });
//     });
// });
