// SPDX-License-Identifier: GPL-3.0-only
// Generic bounded scalar tensor interpreter. No named neural rules, I/O or loading.
using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace NeuralRuntime;

public static class Machine
{
    const double IntermediateLimit=1e13;
    sealed record Tensor(int[] Shape,double[] Data);
    static Exception Fail(string text)=>new InvalidDataException(text);
    static void Fields(JsonObject value,params string[] fields)
    {
        if(!value.Select(p=>p.Key).Order().SequenceEqual(fields.Order()))throw Fail("Unexpected object fields");
    }
    static int Integer(JsonNode? value,int low,int high)
    {
        if(value is not JsonValue v||!v.TryGetValue<int>(out int result)||result<low||result>high)throw Fail("Integer value out of range");
        return result;
    }
    static double Number(JsonNode? value)
    {
        if(value is not JsonValue v||!v.TryGetValue<string>(out string? text)||text.Length>64||
           !double.TryParse(text,NumberStyles.Float,CultureInfo.InvariantCulture,out double result)||
           !double.IsFinite(result)||Math.Abs(result)>1e6)throw Fail("Expected bounded finite decimal string");
        return result;
    }
    static Tensor Bounded(Tensor value,int n)
    {
        bool shape=value.Shape.Length==0||(value.Shape.Length==1&&value.Shape[0] is >=1 and <=9)||
            (value.Shape.Length==2&&value.Shape[0]==n&&value.Shape[1]==n)||
            (value.Shape.Length==3&&value.Shape[0]==n&&value.Shape[1]==n&&value.Shape[2] is >=1 and <=9);
        if(!shape||value.Data.Length>n*n*9||value.Data.Any(v=>!double.IsFinite(v)||Math.Abs(v)>IntermediateLimit))throw Fail("Tensor shape or intermediate range exceeded");
        return value;
    }
    static Tensor Binary(Tensor a,Tensor b,Func<double,double,double> operation)
    {
        bool sa=a.Shape.Length==0,sb=b.Shape.Length==0;
        if(!sa&&!sb&&!a.Shape.SequenceEqual(b.Shape))throw Fail("Binary tensor shapes differ");
        int[] shape=sa?b.Shape:a.Shape;
        var data=new double[Math.Max(a.Data.Length,b.Data.Length)];
        for(int i=0;i<data.Length;i++)data[i]=operation(a.Data[sa?0:i],b.Data[sb?0:i]);
        return new(shape,data);
    }
    static (double[] State,double[] Probabilities) Step(JsonObject program,JsonObject model,double[] state,int n)
    {
        Fields(program,"schema","code");Fields(model,"weights","bias");
        if(program["schema"]?.GetValue<string>()!="nnf-graph/2"||program["code"] is not JsonArray code||code.Count is <1 or >32)throw Fail("Unsupported or excessive graph");
        if(model["weights"] is not JsonArray weights||weights.Count is <1 or >9)throw Fail("Model requires 1..9 weights");
        double[] parameters=weights.Select(Number).ToArray();double bias=Number(model["bias"]);
        var registers=new Tensor?[16];
        int Reg(JsonNode? x)=>Integer(x,0,15);
        Tensor Get(JsonNode? x)=>registers[Reg(x)]??throw Fail("Uninitialized register");
        var arities=new Dictionary<string,int>{{"STATE",2},{"PARAM",3},{"CONST",3},{"GATHER",4},{"DOT",4},
            {"ADD",4},{"MUL",4},{"SIGMOID",3},{"GE",4},{"RETURN",3}};
        for(int pc=0;pc<code.Count;pc++)
        {
            if(code[pc] is not JsonArray instruction||instruction.Count==0||instruction[0] is not JsonValue tag||
               !tag.TryGetValue<string>(out string? op)||!arities.TryGetValue(op,out int length)||instruction.Count!=length)throw Fail("Unknown graph opcode or arity");
            if(op=="RETURN")
            {
                if(pc!=code.Count-1)throw Fail("RETURN must be last");
                var candidate=Get(instruction[1]);var probability=Get(instruction[2]);
                if(!candidate.Shape.SequenceEqual(new[]{n,n})||candidate.Data.Any(x=>x!=0&&x!=1))throw Fail("RETURN state must be a binary grid");
                if(!probability.Shape.SequenceEqual(new[]{n,n})||probability.Data.Any(x=>x<0||x>1))throw Fail("RETURN probabilities must be a grid in [0,1]");
                return((double[])candidate.Data.Clone(),(double[])probability.Data.Clone());
            }
            int destination=Reg(instruction[1]);Tensor result;
            switch(op)
            {
                case "STATE":result=new(new[]{n,n},(double[])state.Clone());break;
                case "PARAM":
                    string parameter=instruction[2]?.GetValue<string>()??"";
                    result=parameter=="weights"?new(new[]{parameters.Length},(double[])parameters.Clone()):parameter=="bias"?new(Array.Empty<int>(),new[]{bias}):throw Fail("Unknown model parameter");break;
                case "CONST":result=new(Array.Empty<int>(),new[]{Number(instruction[2])});break;
                case "GATHER":
                {
                    var source=Get(instruction[2]);
                    if(!source.Shape.SequenceEqual(new[]{n,n})||instruction[3] is not JsonArray offsets||offsets.Count is <1 or >9)throw Fail("Invalid GATHER grid/offsets");
                    var shifts=new List<(int Y,int X)>();
                    foreach(var item in offsets)
                    {
                        if(item is not JsonArray pair||pair.Count!=2)throw Fail("Invalid GATHER offset");
                        shifts.Add((Integer(pair[0],-2,2),Integer(pair[1],-2,2)));
                    }
                    int k=shifts.Count;var data=new double[n*n*k];
                    for(int y=0;y<n;y++)for(int x=0;x<n;x++)for(int c=0;c<k;c++)
                    {int yy=y+shifts[c].Y,xx=x+shifts[c].X;data[(y*n+x)*k+c]=yy>=0&&yy<n&&xx>=0&&xx<n?source.Data[yy*n+xx]:0;}
                    result=new(new[]{n,n,k},data);break;
                }
                case "DOT":
                {
                    var a=Get(instruction[2]);var b=Get(instruction[3]);
                    if(a.Shape.Length!=3||b.Shape.Length!=1||a.Shape[2]!=b.Shape[0])throw Fail("DOT feature/weight mismatch");
                    int k=b.Data.Length;var data=new double[n*n];
                    for(int p=0;p<data.Length;p++)
                    {
                        // Compensated accumulation keeps parity near mixed-sign sums.
                        double sum=0,correction=0;
                        for(int c=0;c<k;c++){double product=a.Data[p*k+c]*b.Data[c],next=sum+product;correction+=Math.Abs(sum)>=Math.Abs(product)?(sum-next)+product:(product-next)+sum;sum=next;}
                        data[p]=sum+correction;
                    }
                    result=new(new[]{n,n},data);break;
                }
                case "ADD":result=Binary(Get(instruction[2]),Get(instruction[3]),(a,b)=>a+b);break;
                case "MUL":result=Binary(Get(instruction[2]),Get(instruction[3]),(a,b)=>a*b);break;
                case "GE":result=Binary(Get(instruction[2]),Get(instruction[3]),(a,b)=>a>=b?1:0);break;
                case "SIGMOID":
                {var source=Get(instruction[2]);result=new(source.Shape,source.Data.Select(x=>1/(1+Math.Exp(-Math.Clamp(x,-60,60)))).ToArray());break;}
                default:throw Fail("Unsupported graph opcode");
            }
            registers[destination]=Bounded(result,n);
        }
        throw Fail("Graph requires RETURN");
    }
    public static string Execute(string json)
    {
        if(json.Length>24000)throw Fail("Payload byte limit");
        var job=JsonNode.Parse(json,documentOptions:new JsonDocumentOptions{MaxDepth=24}) as JsonObject??throw Fail("Invalid image job");
        Fields(job,"schema","program","model","state","tick","steps","runtime");
        if(job["schema"]?.GetValue<string>()!="nnf-native/1"||job["program"] is not JsonObject program||job["model"] is not JsonObject model)throw Fail("Unsupported native image");
        int tick=Integer(job["tick"],0,1000000),steps=Integer(job["steps"],1,24);
        if(tick+steps>1000000)throw Fail("Tick range exceeded");
        if(job["state"] is not JsonArray rows||rows.Count is <8 or >64)throw Fail("State size must be 8..64");
        int n=rows.Count;var state=new double[n*n];
        for(int y=0;y<n;y++)
        {
            if(rows[y] is not JsonArray row||row.Count!=n)throw Fail("State must be square");
            for(int x=0;x<n;x++)state[y*n+x]=Integer(row[x],0,1);
        }
        double[] probabilities=Array.Empty<double>();
        for(int i=0;i<steps;i++)(state,probabilities)=Step(program,model,state,n);
        var next=new JsonArray();var probabilityRows=new JsonArray();
        for(int y=0;y<n;y++)
        {
            var row=new JsonArray();var p=new JsonArray();
            for(int x=0;x<n;x++){row.Add((int)state[y*n+x]);p.Add(probabilities[y*n+x]);}
            next.Add(row);probabilityRows.Add(p);
        }
        job["state"]=next;job["tick"]=tick+steps;job["steps"]=1;
        return new JsonObject{{"value",job},{"probabilities",probabilityRows},{"steps_executed",steps},
            {"active_cells",(int)state.Sum()},{"execution_class","IMAGE_CARRIED_DOTNET_TENSOR_VM"}}.ToJsonString();
    }
}
