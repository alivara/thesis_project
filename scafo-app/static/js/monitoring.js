
$(document).ready(function() {
    
    var socket = io();

    socket.on('connect', function () {
        //socket.emit('my event', {data: 'I\'m connected!'});
        console.log('Connected...')
    });

    // socket.on('monitor_delay_elapsed', function (delay) {
    //     $("#monitor-status-bar").removeClass("alert-info");
    //     $("#monitor-status-bar").addClass("alert-danger");
    //     $("#monitor-status-bar").text("Evaluation in progress");
    // });

    socket.on('monitor_executed', function (result) {

        $("#bumps-result").removeClass("d-none");
        if (result.bumps_detected == true) {
            $("#bbumps-description").text(" Diformation Plot @" + result.capture_time);
            
            var imgurl = result.jac_plot_path + '?' + new Date().getTime(); // force img refresh
            $("#bumps-result img").attr("src", imgurl);
            $("#bumps-result .form-results-img").show();
        } else {
            $("#bumps-description").text("No bumps detected @" + result.capture_time);
            $("#bumps-result .form-results-img").hide();
        }

        $("#shifting-result").removeClass("d-none");
        $("#shifting-description").html("");
        if (result.shifting_detected == true) {
            for (let i = 0; i < result.shifting_results.length; i++) {
                const element = result.shifting_results[i];
                $("#shifting-description").append(element);
                $("#shifting-description").append("<br/>");
            }

            var imgurl = result.shifting_image_path + '?' + new Date().getTime(); // force img refresh
            $("#shifting-result img").attr("src", imgurl);
            $("#shifting-result .form-results-img").show();
        } else {
            $("#shifting-description").text("No shifting detected");
            $("#shifting-result .form-results-img").hide();
        }

        $("#monitor-status-bar").removeClass("alert-danger");
        $("#monitor-status-bar").addClass("alert-info");
        $("#monitor-status-bar").text("Monitoring...");

    });


    socket.on('monitor_status', function (status) {
        
        if (status.includes("Stopped")) {
            $("#monitor-status-bar").addClass("alert-danger");

        } else if (status.includes("Evaluation")) {
            $("#monitor-status-bar").addClass("alert-danger");

        } else {
            $("#monitor-status-bar").addClass("alert-info");
        }
        
        $("#monitor-status-bar").text(status);

    });



});


