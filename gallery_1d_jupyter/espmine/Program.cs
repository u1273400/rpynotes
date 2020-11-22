using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using Slack.Webhooks;

namespace espmine
{
    class Slackbot
    {
        private const string root = "c:/Users/User/Dropbox/rtmp/src/python/notebooks/espnet/egs/an4/asr1s/exp/train_nodev_pytorch_train_mtlalpha1.0";
        private int msg_counter=0;

        private ProcessStartInfo tailInfo  = new ProcessStartInfo
                {
                    FileName = @"C:\Program Files\Git\usr\bin\tail",
                    Arguments = $"-n 1 {root}/train.log",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

        public static void Main()
        {
            Run();
        }

        private static void Run()
        {
            Slackbot bot=new Slackbot();

            string[] args = Environment.GetCommandLineArgs();;
            // If a directory is not specified, exit program.
            if (args.Length != 2)
            {
                // Display the proper way to call the program.
                Console.WriteLine("Warnging Usage: Watcher.exe (espnet subd)");
                //return;file:///C:/Users/jack/desktop/list-of-items.txt
            }
            bot.setupFolder();

        }
        private void setupFolder(){
                        // Create a new FileSystemWatcher and set its properties.
            using (FileSystemWatcher watcher = new FileSystemWatcher())
            {
                watcher.Path = root;
                // watcher.Path = args[1];

                // Watch for changes in LastAccess and LastWrite times, and
                // the renaming of files or directories.
                watcher.NotifyFilter = NotifyFilters.LastAccess
                                    | NotifyFilters.LastWrite
                                    | NotifyFilters.FileName
                                    | NotifyFilters.DirectoryName;

                // Only watch train.log.
                watcher.Filter = "*.log";

                // Add event handlers.
                watcher.Changed += OnChanged;
                watcher.Created += OnChanged;
                watcher.Deleted += OnChanged;
                watcher.Renamed += OnRenamed;

                // Begin watching.
                watcher.EnableRaisingEvents = true;

                // Wait for the user to quit the program.
                Console.WriteLine("Press 'q' to quit the sample.");
                while (Console.Read() != 'q') ;
            }

        }
        private void postMessage(string message){
            var url = "https://hooks.slack.com/services/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";

            var slackClient = new SlackClient(url);

            var slackMessage = new SlackMessage
            {
                Channel = "@jesusluvsu",
                Text = message,
                IconEmoji = Emoji.CreditCard,
                Username = "espnet research"
            };

            slackClient.Post(slackMessage);            
        }
        private Task<int> RunTailAsync()
        {
            var tcs = new TaskCompletionSource<int>();

            var process = new Process
            {
                StartInfo = tailInfo,
                EnableRaisingEvents = true
            };

            process.Exited += (sender, args) =>
            {
                tcs.SetResult(process.ExitCode);
                process.Dispose();
            };

            process.ErrorDataReceived += (sender, args)=>
            {
                string err = process.StandardError.ReadToEnd();
                Console.WriteLine(err);
            };

            process.OutputDataReceived += (sender, args)=>{
                Console.WriteLine("data received: "+args.Data);
                postMessage(args.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            return tcs.Task;
        }

        private void tail(){
        try{
            var not_good=File.ReadAllLines($"{root}/train.log");
            var n=not_good.Length;
            Console.WriteLine($"{n} lines read");
            if(n>0 && n%2==0)postMessage(not_good[n-1]);
        }catch(IOException x){
            Console.WriteLine($"{x.Message} retrying..");
            Thread.Sleep(2000);
            if(++msg_counter==10)return;
            tail();
        }finally{
            Console.WriteLine("Using Tasync..");
            RunTailAsync();
            Thread.Sleep(200);
        }
       }

        // Define the event handlers.
        private void OnChanged(object source, FileSystemEventArgs e){
            // Specify what is done when a file is changed, created, or deleted.
            Console.WriteLine($"File: {e.FullPath} {e.ChangeType}");
            msg_counter=0;
            this.tail();
        }
        private void OnRenamed(object source, RenamedEventArgs e) =>
            // Specify what is done when a file is renamed.
            Console.WriteLine($"File: {e.OldFullPath} renamed to {e.FullPath}");

    }
}
