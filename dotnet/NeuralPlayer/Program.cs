// SPDX-License-Identifier: GPL-3.0-only
using System.Text.Json.Nodes;

namespace NeuralPlayer;

internal static class Program
{
    public static (Bitmap Image,JsonObject Receipt) Refresh(Bitmap image)
    {
        var job=Carrier.Decode(image);var receipt=Carrier.Execute(job);var next=Display.Render(receipt["value"]!.AsObject(),receipt["probabilities"]!.AsArray());
        if(!Carrier.Canonical(Carrier.Decode(next)).SequenceEqual(Carrier.Canonical(receipt["value"])))throw new InvalidDataException("Refresh payload mismatch");
        return(next,receipt);
    }
    [STAThread]
    static int Main(string[] args)
    {
        try
        {
            ApplicationConfiguration.Initialize();
            if(args.Length>=2&&args[0]=="--step")
            {
                if(args.Length!=2&&(args.Length!=4||args[2]!="--out"))throw new ArgumentException("Use --step image [--out new.tiff|new.gif]");
                using var source=Carrier.Open(args[1]);var(next,receipt)=Refresh(source);
                using(next){if(args.Length==4)Carrier.Save(next,args[3]);Console.WriteLine(receipt.ToJsonString());}return 0;
            }
            bool smoke=args.Length>=2&&args[0]=="--smoke-ui";
            string? path=smoke?args[1]:args.FirstOrDefault();
            if(path is null)
            {
                string example=Path.Combine(AppContext.BaseDirectory,"examples","or.tiff");
                if(File.Exists(example))path=example;
                else{using var picker=new OpenFileDialog{Title="Open a neural image program",Filter="Image programs|*.tiff;*.tif;*.gif;*.png"};if(picker.ShowDialog()!=DialogResult.OK)return 0;path=picker.FileName;}
            }
            string? smokeExport=smoke&&args.Length==4&&args[2]=="--export-dir"?args[3]:null;
            using var window=new Player(path,smoke,smokeExport);Application.Run(window);return window.Failed?1:0;
        }
        catch(Exception error){Console.Error.WriteLine("Carrier rejected: "+error.Message);return 1;}
    }
}

internal static class Display
{
    public static Bitmap Render(JsonObject job,JsonArray? probabilities=null)
    {
        using var visible=new Bitmap(Carrier.Width,Carrier.Height);
        using(var g=Graphics.FromImage(visible))
        {
            g.Clear(Color.FromArgb(16,26,43));using var title=new Font("Segoe UI",24,FontStyle.Bold);using var text=new Font("Segoe UI",12);
            using var green=new SolidBrush(Color.FromArgb(104,238,174));using var muted=new SolidBrush(Color.FromArgb(174,192,215));
            g.DrawString("Neuron Noodle Field 0.4.0",title,Brushes.White,24,18);
            g.DrawString("Runtime + graph + model + current state come from this image",text,muted,26,62);
            var rows=job["state"]!.AsArray();int n=rows.Count,active=0;float cell=320f/n;
            for(int y=0;y<n;y++)for(int x=0;x<n;x++)
            {
                int bit=rows[y]![x]!.GetValue<int>();active+=bit;
                g.FillRectangle(bit==1?green:Brushes.MidnightBlue,35+x*cell,116+y*cell,cell,cell);
                double p=probabilities is null?bit:probabilities[y]![x]!.GetValue<double>();
                using var heat=new SolidBrush(Color.FromArgb((int)(35+220*p),(int)(35+130*p),(int)(75+80*(1-p))));
                g.FillRectangle(heat,500+x*cell,116+y*cell,cell,cell);
            }
            g.DrawString($"Binary field: {active} active    Tick: {job["tick"]}",text,Brushes.White,35,86);
            g.DrawString(probabilities is null?"Saved field (probabilities recalculate on execution)":"Computed probability",text,Brushes.White,500,86);
            g.DrawString("Space: run / pause    Right: execute image    O: open    E: export    R: reload",text,muted,26,450);
            g.DrawString("Generic CPU tensor VM recovered from verified raster bytes; no optical hardware",text,muted,26,480);
        }
        return Carrier.Encode(visible,job);
    }
}

internal sealed class Player:Form
{
    readonly PictureBox picture=new(){Dock=DockStyle.Fill,SizeMode=PictureBoxSizeMode.Zoom};
    readonly Label status=new(){Dock=DockStyle.Bottom,Height=30};
    readonly System.Windows.Forms.Timer timer=new(){Interval=350};
    Bitmap current;string path;bool running;readonly bool smoke;readonly string? smokeExport;int smokeTicks;
    public bool Failed{get;private set;}
    public Player(string input,bool smoke,string? smokeExport=null)
    {
        this.smoke=smoke;this.smokeExport=smokeExport;path=input;current=Carrier.Open(path);Text="Neuron Noodle Field 0.4.0 — interpreter in pixels";
        ClientSize=new Size(960,790);MinimumSize=new Size(640,520);KeyPreview=true;
        var bar=new FlowLayoutPanel{Dock=DockStyle.Top,Height=40};
        void Button(string name,Action callback){var b=new Button{Text=name,AutoSize=true};b.Click+=(_,_)=>Guard(callback);bar.Controls.Add(b);}
        Button("Execute image",Step);Button("Run / pause",()=>running=!running);Button("Reload",Reload);Button("Open image",Open);Button("Export TIFF + GIF",Export);
        Controls.Add(picture);Controls.Add(status);Controls.Add(bar);picture.Image=current;Status();
        timer.Tick+=(_,_)=>
        {
            if(!running)return;Guard(Step);
            if(smoke&&(Failed||++smokeTicks>=3))
            {
                running=false;
                if(!Failed&&smokeExport is not null)Guard(()=>{Directory.CreateDirectory(smokeExport);ExportTo(Path.Combine(smokeExport,"ui.tiff"));});
                if(!Failed)Console.WriteLine(new JsonObject{{"ui_constructed",true},{"refresh_executed",true},{"automatic_timer_refreshes",smokeTicks},{"keyboard_handler_exercised",true},{"export_callback_exercised",smokeExport is not null},{"final_tick",Carrier.Decode(current)["tick"]!.DeepClone()}}.ToJsonString());
                Close();
            }
        };timer.Start();
        Shown+=(_,_)=>{if(smoke){Message m=Message.Create(Handle,0x100,(nint)Keys.Right,0);ProcessCmdKey(ref m,Keys.Right);ProcessCmdKey(ref m,Keys.Space);if(Failed)Close();}};
    }
    void Guard(Action action){try{action();Failed=false;Status();}catch(Exception e){running=false;Failed=true;status.Text="Stopped: "+e.Message;if(smoke)Console.Error.WriteLine(e.Message);}}
    void Status(){if(!Failed){var job=Carrier.Decode(current);status.Text=$"Tick {job["tick"]} | {(running?"Running":"Paused")} | Space runs; Right executes; O opens another model; E exports a continuation image.";}}
    void Replace(Bitmap next){picture.Image=next;current.Dispose();current=next;}
    void Step(){var(next,_)=Program.Refresh(current);Replace(next);}
    void Reload(){Replace(Carrier.Open(path,true));running=false;}
    void Open(){running=false;using var picker=new OpenFileDialog{Filter="Neural image programs|*.tiff;*.tif;*.gif;*.png"};if(picker.ShowDialog()!=DialogResult.OK)return;var next=Carrier.Open(picker.FileName);path=picker.FileName;Replace(next);}
    void Export()
    {
        running=false;using var picker=new SaveFileDialog{Filter="TIFF image program|*.tiff",FileName="neural-"+DateTime.Now.ToString("yyyyMMdd-HHmmss")+".tiff"};
        if(picker.ShowDialog()!=DialogResult.OK)return;ExportTo(picker.FileName);
    }
    void ExportTo(string path)
    {
        string gif=Path.ChangeExtension(path,"gif");
        if(File.Exists(path)||File.Exists(gif))throw new IOException("Use new filenames; existing output is preserved");
        Carrier.Save(current,path);Carrier.Save(current,gif);
        using var a=Carrier.Open(path);using var b=Carrier.Open(gif);
        if(!Carrier.Canonical(Carrier.Decode(a)).SequenceEqual(Carrier.Canonical(Carrier.Decode(current)))||!Carrier.Canonical(Carrier.Decode(b)).SequenceEqual(Carrier.Canonical(Carrier.Decode(current))))throw new InvalidDataException("Export verification failed");
    }
    protected override bool ProcessCmdKey(ref Message msg,Keys key)
    {
        if(key==Keys.Space){Guard(()=>running=!running);return true;}
        Action? action=key switch{Keys.Right=>Step,Keys.R=>Reload,Keys.O=>Open,Keys.E=>Export,_=>null};
        if(action is not null){Guard(action);return true;}return base.ProcessCmdKey(ref msg,key);
    }
    protected override void Dispose(bool disposing){if(disposing){timer.Dispose();picture.Image=null;current.Dispose();}base.Dispose(disposing);}
}
