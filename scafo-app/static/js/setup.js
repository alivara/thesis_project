
$(document).ready(function() {
    
    $("#n_dipole").on('change', function (e) {
        $("#n_dipole-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#br").on('change', function (e) {
        $("#br-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#ro").on('change', function (e) {
        $("#ro-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#fo").on('change', function (e) {
        $("#fo-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#zo").on('change', function (e) {
        $("#zo-label").text(this.value);
        $(this.form).trigger('submit');
    });

    $("#Config-1").on('change', function (e) {
        $('input[id= br]').val(143);
        $('input[id= ro]').val(32);
        $('input[id= fo]').val(52);
        $('input[id= zo]').val(9);

        $('span[id= br-label]').text(143);
        $('span[id= ro-label]').text(32);
        $('span[id= fo-label]').text(52);
        $('span[id= zo-label]').text(9);
        $(this.form).trigger('submit');
    });
    
    $("#Config-2").on('change', function (e) {
        $('input[id= br]').val(255);
        $('input[id= ro]').val(0);
        $('input[id= fo]').val(100);
        $('input[id= zo]').val(100);

        $('span[id= br-label]').text(255);
        $('span[id= ro-label]').text(0);
        $('span[id= fo-label]').text(84);
        $('span[id= zo-label]').text(44);
        $(this.form).trigger('submit');

    });
    
    $("#Config-3").on('change', function (e) {
        $('input[id= br]').val(255);
        $('input[id= ro]').val(6);
        $('input[id= fo]').val(87);
        $('input[id= zo]').val(63);

        $('span[id= br-label]').text(255);
        $('span[id= ro-label]').text(6);
        $('span[id= fo-label]').text(87);
        $('span[id= zo-label]').text(63);
        $(this.form).trigger('submit');
    });

    
});


//plugin bootstrap minus and plus
jQuery(document).ready(function(){
    // This button will increment the value
    $('.qtyplus').click(function(e){
        // Stop acting like a button
        e.preventDefault();
        // Get the field name
        fieldName = $(this).attr('field');
        // Get its current value
        var currentVal = parseInt($('input[name='+fieldName+']').val());
        // If is not undefined
        if (!isNaN(currentVal)) {
            // Increment
            $('input[name='+fieldName+']').val(currentVal + 1);
            $('span[id='+fieldName+'-label]').text(currentVal + 1);
            $(this.form).trigger('submit');
        } else {
            // Otherwise put a 0 there
            $('input[name='+fieldName+']').val(0);
            $('span[id='+fieldName+'-label]').text(0);
            $(this.form).trigger('submit');
        }
    });
    // This button will decrement the value till 0
    $(".qtyminus").click(function(e) {
        // Stop acting like a button
        e.preventDefault();
        // Get the field name
        fieldName = $(this).attr('field');
        // Get its current value
        var currentVal = parseInt($('input[name='+fieldName+']').val());
        // If it isn't undefined or its greater than 0
        if (!isNaN(currentVal) && currentVal > 0) {
            // Decrement one
            $('input[name='+fieldName+']').val(currentVal - 1);
            $('span[id='+fieldName+'-label]').text(currentVal - 1);
            $(this.form).trigger('submit');
        } else {
            // Otherwise put a 0 there
            $('input[name='+fieldName+']').val(0);
            $('span[id='+fieldName+'-label]').text(0);
            $(this.form).trigger('submit');
        }
    });
});
