(function($){

    var defaultSettings = {
        size: 0,
        quantity: 10,
        dotSize: 6,
        enableReverse: true,
        waitMovementIncrementer: 1,
        light: false,
        fullScreen: false,
        speed: 100,
        circleCount: 1,
        tailPercent: 0.25
    };
    
    $.waiting = function(el, options){
        var base = $(el).data("waiting");
        if(typeof base === 'undefined'){
          base = this;
            // Access to jQuery and DOM versions of element
            base.el = el;
            base.$el = $(el);
            // Add a reverse reference to the DOM object
            base.$el.data("waiting",base);
        }
        if(typeof options === "string"){
            switch(options){
                case 'pause':
                    base.pause();
                    break;
                case 'play':
                    base.play();
                    break;
                case 'done':
                    base.done();
                    break;
            }
        }
        
        base.init = function(){
            base.userSettings = $.extend({},defaultSettings, options);
            base.position = {top:0,left:0,height:0,width:0};
            base.originalPosition = base.position;
            base.intervalID = 0;
            base.resizeIntervalID = 0;
            base.createWaiting();
        };
        base.pause = function(){
          clearInterval(base.intervalID);
        };
        base.play = function(){
          base.pause();
          base.intervalID = setInterval(function(){
            base.waitingLoopColorChange();
          }, base.userSettings.speed);
        };
        
        base.done = function(){
          if(typeof base.$mask !== 'undefined'){
            base.pause();
            base.$mask.remove();
            base.$el.removeData("waiting");
          }
        };
        base.createWaiting = function(){
          base.done();
          if(typeof base.$mask === 'undefined'){
            base.$mask = $('<div style="position:absolute;z-index:9000;background-color:#' + (base.userSettings.light ? '9090a6' : '222') + ';filter:alpha(opacity=50);opacity:0.5;"></div>').appendTo('body');
            if(base.userSettings.enableReverse){
              base.$mask.click(function(){
                base.userSettings.waitMovementIncrementer *= -1;
              });
            }
          }
          if(typeof base.$wait === 'undefined'){
            base.$wait = $('<div class="ringHolder" style="z-index:9001;position:relative;"></div>').appendTo(base.$mask);
          }
          base.waitSizing();
          
          base.pause();
          clearInterval(base.resizeIntervalID);
          
          base.play();
          
          base.resizeIntervalID = setInterval(function(){
            base.waitingTargetMoved();
          }, 20);
          
          $(window).resize(function(){
            base.waitSizing();
          });
        };
        
        base.waitingTargetMoved = function(){
          var hidden = base.$el.is(':hidden');
          base.$mask.toggle(!hidden);
          if(!hidden){
            base.getLeftTopHeightWidth();
            if(base.position.top !== base.originalPosition.top || base.position.left !== base.originalPosition.left || 
              base.position.height !== base.originalPosition.height || base.position.width !== base.originalPosition.width){
                base.waitSizing();
              }
          }
        };
        
        base.getLeftTopHeightWidth = function(){
          if(base.userSettings.fullScreen){
            base.position.height = $(window).outerHeight();
            base.position.width = $(window).outerWidth();
            base.position.top = window.pageYOffset;
            base.position.left = window.pageXOffset;
          } else {
            base.position.height = base.$el.outerHeight();
            base.position.width = base.$el.outerWidth();
            if(base.$el.offset() !== undefined){
              var offset = base.$el.offset();
              base.position.top = offset.top;
              base.position.left = offset.left;
            }
          }
          return base.position;
        };
        base.waitSizing = function(){
            var size = base.userSettings.size;
            var dotSize = base.userSettings.dotSize;
              
            base.getLeftTopHeightWidth();
            base.resetPosition();
            var maskHeight = base.position.height;
            var maskWidth = base.position.width;
            if(size === 0 || (size > (base.position.height - dotSize) || size > (base.position.width - dotSize))){
                var boxSize = base.userSettings.size;
                if(base.userSettings.fullScreen){
                    maskHeight = $('body').outerHeight();
                    maskWidth = $('body').outerWidth();
                    boxSize = Math.min(tempHeight, tempWidth) - dotSize;
                } else {
                    boxSize = (base.position.height < base.position.width ? base.position.height : base.position.width) - dotSize;
                }
                size = boxSize;
            }
            base.$mask.height(maskHeight).width(maskWidth).offset({top:base.position.top, left:base.position.left});
            base.$wait.height(size).width(size).offset({top:(base.position.top + (base.position.height/2 - size/2)), left:(base.position.left+ (base.position.width/2 - size/2))}).html('');
            base.drawCircles(size);
        };
        
        base.drawCircles = function(size){
          if(base.$wait.children().filter('.ring').length === 0){
            var dotSize = base.userSettings.dotSize;
            var quantity = base.userSettings.quantity;
            var circleCount = base.userSettings.circleCount;
            base.loopMax = (quantity - 1);
            var halfDot = dotSize / 2;
            for(var circ = 0; circ < circleCount; circ++){
              for(var ndx = 0; ndx < quantity; ndx++){
                var cxcy = size / 2;
                var angle = ((360 / quantity) * (ndx + 1)) * Math.PI / 180;
                var radius = cxcy - ((circ) * (cxcy / circleCount));
                var x = (cxcy + radius * Math.cos(angle)) - halfDot;
                var y = (cxcy + radius * Math.sin(angle)) - halfDot;
                base.$wait.append('<div loopOrder="' + ndx + '" class="ring" style="filter:alpha(opacity=100); opacity:1; left:' + x + 'px; top:' + y + 'px; height:' + dotSize + 'px; width:' + dotSize + 'px; position:absolute; background-color:#777; -moz-border-radius:' + dotSize + 'px; -webkit-border-radius:' + dotSize + 'px; border-radius:' + dotSize + 'px;"></div>');
              }
            }
          }
        };
        
        base.waitingLoopColorChange = function(){
          base.$wait.children('div').each(function(i,e){
            $this = $(this);
            var order = 1 * $this.attr('loopOrder');
             var incrementer = (base.userSettings.waitMovementIncrementer * 1) > 0 ? 1 : -1;
            order += incrementer;
            if (order > base.loopMax){
              order = 0;
            }
            if (order < 0) {
              order = base.loopMax;
            }
            $this.attr('loopOrder', order);
            var tailPercent = base.userSettings.tailPercent * 1;
            if (tailPercent > 1)
              tailPercent = 1;
            var tail = (base.loopMax * tailPercent);
            var tailSections = (tail < 4 ? 1 : (tail / 4));
            var upwards = incrementer > 0;
            var useLight = base.userSettings.light;
            var color = useLight ? '#1e64ae' : '#666';
            if (order <= tailSections){
              color = (upwards ? (useLight ? '#fafcfe' : '#ddd') : (useLight ? '#3e84ce' : '#888'));
            } else if (order <= tailSections * 2){
              color = (upwards ? (useLight ? '#bcd4ee' : '#bbb') : (useLight ? '#7dacde' : '#999'));
            } else if (order <= tailSections * 3){
              color = (upwards ? (useLight ? '#7dacde' : '#999') : (useLight ? '#bcd4ee' : '#bbb'));
            } else if (order <= tailSections * 4){
              color = (upwards ? (useLight ? '#3e84ce' : '#888') : (useLight ? '#fafcfe' : '#ddd'));
            }
            $this.css("backgroundColor" ,color);
          });
        };
        
        base.resetPosition = function(){
          base.originalPosition.top = base.position.top;
          base.originalPosition.left = base.position.left;
          base.originalPosition.height = base.position.height;
          base.originalPosition.width = base.position.width;
        };
        
        if(typeof options !== "string")
            base.init();
    };
    
    
    
    $.fn.waiting = function(options){
        return this.each(function(){
            (new $.waiting(this, options));
        });
    };
    
})(jQuery);
