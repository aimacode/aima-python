var latest_output_area ="NONE"; // Jquery object for the DOM element of output area which was used most recently

function handle_output(out, block){
    var output = out.content.data["text/html"];
    latest_output_area.html(output);
}

function handle_click(canvas,coord) {
	console.log(canvas,coord);
	latest_output_area = $(canvas).parents('.output_subarea');
	$(canvas).parents('.output_subarea')
    var world_object_name = canvas.dataset.world_name;
    var command = world_object_name + ".handle_click(" + JSON.stringify(coord) + ")";
    console.log("Executing Command: " + command);
    var kernel = IPython.notebook.kernel;
    var callbacks = { 'iopub' : {'output' : handle_output}};
    kernel.execute(command,callbacks);
};


function generateGridWorld(state,size,elements)
{
	// Declaring array to store image object
	var $imgArray = new Object(), hasImg=false;
	// Loading images LOOP
	$.each(elements, function(i, val) {
	    // filtering for type img
	    if(val["type"]=="img") {
	    	// setting image load
	    	hasImg = true;
		    $imgArray[i] = $('<img />').attr({height:size,width:size,src:val["source"]}).data({name:i,loaded:false}).load(function(){
		    	// Check for all image loaded
		    	var execute=true;
		    	$(this).data("loaded",true);
		    	$.each($imgArray, function(i, val) {
		    		if(!$(this).data("loaded")) {
		    			execute=false;
		    			// exit on unloaded image
		    			return false;
		    		}
		    	});	
		    	if (execute) {
		    		// Converting loaded image to canvas covering block size.
		    		$.each($imgArray, function(i, val) {
		    			$imgArray[i] = $('<canvas />').attr({width:size,height:size}).get(0);
		    			$imgArray[i].getContext('2d').drawImage(val.get(0),0,0,size,size);
		    		});	
		    		// initialize the world
		    		initializeWorld();	
		    	}
		    });
	    }
	});

	if(!hasImg) {
		initializeWorld();
	}

	function initializeWorld(){
		var $parentDiv = $('div.map-grid-world');
		// remove object reference
		$('div.map-grid-world').removeClass('map-grid-world');
		// get some info about the canvas
		var row = state.length;
		var column = state[0].length;
		var canvas = $parentDiv.find('canvas').get(0);
		var ctx = canvas.getContext('2d');
		canvas.width =  size * column;
		canvas.height = size * row;
		
		//Initialize previous positions
		for(var i=0;i<row;++i) {
			for (var j = 0; j < column;++j) {
				if(elements[state[i][j]["val"]]["type"]=="color") {
					if( state[i][j]["val"] == "default") {
						blockCreate('black',elements[state[i][j]["val"]]["source"], i, j);
					}
					else {
						blockCreate('transparent',elements[state[i][j]["val"]]["source"], i, j);
					}
				}
				else if(elements[state[i][j]["val"]]["type"]==="img") {
					pattern = ctx.createPattern($imgArray[state[i][j]["val"]], "repeat");
					blockCreate('transparent',pattern, i, j);
				}
			}
		}

		// function to create a block of given parameters
		function blockCreate(borderColor,fillColor,x,y) {
	        ctx.fillStyle = fillColor;
			ctx.fillRect(y * size, x * size, size, size);
			ctx.lineWidth=2;
	        ctx.strokeStyle = borderColor;
			ctx.strokeRect((y * size)+1, (x * size)+1, size-2, size-2);
		}

		// click event, using jQuery for cross-browser convenience
		$(canvas).click(function(e) {
		    // calculate grid square numbers
		    var gy = ~~ (e.offsetX / size);
		    var gx = ~~ (e.offsetY/ size);
		    //handle click send data
		    var coord = new Array();
		    coord.push(gx,gy)
		    handle_click(canvas,coord)
		});

		// Display tooltip and change on mousemove and mouseout
		var nameElement = $parentDiv.find('span');
		$(canvas).mousemove(function(e) {
		    //calculate grid square numbers
		    var gy = ~~ (e.offsetX / size);
		    var gx = ~~ (e.offsetY/ size);
		    // updating tooltip
		    if( gx>=0 && gx<row && gy>=0 && gy<column ) {
		    	nameElement.html(state[gx][gy]["tooltip"]);    
		    }
		}).mouseout(function(){
		    nameElement.html("");    
		});
	};
};

// function call
generateGridWorld(gridArray,size,elements);