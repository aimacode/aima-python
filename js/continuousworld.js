var latest_output_area ="NONE"; // Jquery object for the DOM element of output area which was used most recently
function handle_output(out, block){
    var output = out.content.data["text/html"];
    latest_output_area.html(output);
}
function polygon_complete(canvas, vertices){
    latest_output_area = $(canvas).parents('.output_subarea');
    var world_object_name = canvas.dataset.world_name;
    var command = world_object_name + ".handle_add_obstacle(" + JSON.stringify(vertices) + ")";
    console.log("Executing Command: " + command);
    var kernel = IPython.notebook.kernel;
    var callbacks = { 'iopub' : {'output' : handle_output}};
    kernel.execute(command,callbacks);
}
var canvas , ctx;
function drawPolygon(array) {
    ctx.fillStyle = '#f00';
    ctx.beginPath();
    ctx.moveTo(array[0][0],array[0][1]);
    for(var i = 1;i<array.length;++i)
    {
        ctx.lineTo(array[i][0], array[i][1]);
    }
    ctx.closePath();
    ctx.fill();
}
var pArray = new Array();
function getPosition(obj,event) {
    canvas = obj;
    ctx = canvas.getContext('2d');
    var x = new Number();
    var y = new Number();
    x = event.pageX;
    y = event.pageY;
    x -= $(canvas).offset().left;
    y -= $(canvas).offset().top;
    drawPoint(x,y);
    //draw dot
    if(pArray.length>1)
    {
        drawPoint(pArray[0][0],pArray[0][1]);
    }
    //check overlap
    if(ctx.isPointInPath(x, y) && (pArray.length>1)) {
        //Do something
        drawPolygon(pArray);
        polygon_complete(canvas,pArray);
    }
    else {
        var point = new Array();
        point.push(x,y);
        pArray.push(point);
    }
}
function drawPoint(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI*2);
    ctx.fillStyle = '#00f';
    ctx.fill();
    ctx.closePath();
}
function initalizeObstacles(objects) {
    canvas = $('canvas.main-robo-world').get(0);
    ctx = canvas.getContext('2d');
    $('canvas.main-robo-world').removeClass('main-robo-world');
    for(var i=0;i<objects.length;++i) {
        drawPolygon(objects[i]);
    }
    pArray.length = 0;
}
initalizeObstacles(all_polygons);