function output_callback(out, block){
    console.log(out);
    script = out.content.data['text/html'];
    console.log(script);
    script = script.substr(8, script.length - 17);
    console.log(script);
    eval(script)
}

function click_callback(element, event, varname){
    var rect = element.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    var kernel = IPython.notebook.kernel;
    var exec_str = varname + ".mouse_click(" + String(x) + ", " + String(y) + ")";
    console.log(exec_str);
    kernel.execute(exec_str,{'iopub': {'output': output_callback}}, {silent: false});
}

function rgbToHex(r,g,b){
    var hexValue=(r<<16) + (g<<8) + (b<<0);
    var hexString=hexValue.toString(16);
    //Add 0 padding
    hexString ='#' + Array(7-hexString.length).join('0') + hexString;
    return hexString;
}

function toRad(x){
    return x*Math.PI/180;
}

//Canvas class to store variables
function Canvas(id){
    this.canvas = document.getElementById(id);
    this.ctx = this.canvas.getContext("2d");
    this.WIDTH = this.canvas.width;
    this.HEIGHT = this.canvas.height;
    this.MOUSE = {x:0,y:0};
}

Canvas.prototype.fill = function(r, g, b){
    //Sets the fill color
    this.ctx.fillStyle = rgbToHex(r,g,b);
}

Canvas.prototype.stroke = function(r, g, b){
    //Set the stroke color
    this.ctx.strokeStyle = rgbToHex(r,g,b);
}

Canvas.prototype.strokeWidth = function(w){
    //Set width of the line
    this.ctx.lineWidth = w;
}

Canvas.prototype.rect = function(x, y, w, h){
    this.ctx.fillRect(x,y,w,h);
}

Canvas.prototype.line = function(x1, y1, x2, y2){
    this.ctx.beginPath();
    this.ctx.moveTo(x1, y1);
    this.ctx.lineTo(x2, y2);
    this.ctx.stroke();
}

Canvas.prototype.arc = function(x, y, r, start, stop){
    this.ctx.beginPath();
    this.ctx.arc(x, y, r, toRad(start), toRad(stop));
    this.ctx.stroke();
}

Canvas.prototype.clear = function(){
    this.ctx.clearRect(0, 0, this.WIDTH, this.HEIGHT);
}

Canvas.prototype.test_run = function(){
    var dbg = false;
    if(dbg)
	alert("1");
    this.clear();
    if(dbg)
	alert("2");
    this.fill(0, 200, 0);
    if(dbg)
	alert("3");
    this.rect(this.MOUSE.x, this.MOUSE.y, 100, 200);
    if(dbg)
	alert("4");
    this.stroke(0, 0, 50);
    if(dbg)
	alert("5");
    this.line(0, 0, 100, 100);
    if(dbg)
	alert("6");
    this.stroke(200, 200, 200);
    if(dbg)
	alert("7");
    this.arc(200, 100, 50, 0, 360);
    if(dbg)
	alert("8");
}
