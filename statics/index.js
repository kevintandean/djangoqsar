/**
 * Created by kevintandean on 4/28/15.
 */
(function($) {
    $.fn.onEnter = function(func) {
        this.bind('keypress', function(e) {
            if (e.keyCode == 13) func.apply(this, [e]);
        });
        return this;
     };
})(jQuery);

var get_result = function(data){
        console.log(data);
        $.ajax({
            url: "/result/",
            type: 'POST',
            dataType:'html',
            data: data,
            success: function (data) {
                console.log(data)
                $("#result_holder").html(data);
            }
        });
}

$(document).ready(function () {
    $(document).on('click', '#check', function () {
        $('#result_holder').html("");
       var data = JSON.stringify($('#id_query').val());
        get_result(data)
    });
    $('#id_query').onEnter(function () {
                $('#result_holder').html("");
        var data = JSON.stringify($('#id_query').val());
        get_result(data)
    })
})
