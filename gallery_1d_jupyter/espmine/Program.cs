using System;
using System.Security.Cryptography;
using System.Text;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Threading;
using Slack.Webhooks;
using System.Net;
using System.Net.Mail;
//
namespace espmine
{
    class Slackbot
    {
        //private const string root = "/home/john/src/python/espnet/egs/an4/asr1s/exp/train_nodev_pytorch_train_mtlalpha1.0";
        //private const string root = "/home/john/src/python/espnet/egs/commonvoice/asr1s/exp/";
        //private static string root = "/home/john/src/python/espnet/egs/voxforge/asr1s/exp/";
	private static string log = $"{root}/make_fbank/valid_test_en/make_fbank_pitch_valid_test_en.2.log";
        private const string root = "/home/john/src/python/espnet/egs/commonvoice/asr1s/exp/";
        static string key  = "A!9HHhi%XjjYY4YP2@Nob009X";
        static string apikey  = "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";
	static string sendgrid = "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";
        private int msg_counter=0;

	public void email_send(String msg)
	{
	    MailMessage mail = new MailMessage();
	    SmtpClient SmtpServer = new SmtpClient("smtp.gmail.com");
	    mail.From = new MailAddress("john.alamina@gmail.com");
	    mail.To.Add("john.alamina@hud.ac.uk");
	    mail.Subject = "ESPNET loss";
	    mail.Body = msg;

	    //System.Net.Mail.Attachment attachment;
	    //var loss = new System.Net.Mail.Attachment($"{root}/results/loss.png");
	    //var acc = new System.Net.Mail.Attachment($"{root}tr_it_pytorch_train/results/acc.png");
	    //mail.Attachments.Add(acc);
	    //mail.Attachments.Add(loss);

	    SmtpServer.Port = 587;
            SmtpServer.Credentials = new System.Net.NetworkCredential("john.alamina@gmail.com", Decrypt("LPgmqcM+CA3LKyzqiijbYA=="));
	    SmtpServer.EnableSsl = true;
    
	    SmtpServer.Send(mail);

	}
        private ProcessStartInfo tailInfo  = new ProcessStartInfo
        {
            FileName = @"/usr/bin/tail",
            Arguments = $"-n 1 {root}/train.log",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            CreateNoWindow = true
        };

        public static async Task Main()
        {
            await Run();
        }

        private static async Task Run()
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
	    while(true){
		await bot.tail();
		Thread.Sleep(1000);
	    }
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

        public static string Decrypt(string cipher)
        {
            using (var md5 = new MD5CryptoServiceProvider())
            {
                using (var tdes = new TripleDESCryptoServiceProvider())
                {
                    tdes.Key = md5.ComputeHash(UTF8Encoding.UTF8.GetBytes(key));
                    tdes.Mode = CipherMode.ECB;
                    tdes.Padding = PaddingMode.PKCS7;

                    using (var transform = tdes.CreateDecryptor())
                    {
                        byte[] cipherBytes = Convert.FromBase64String(cipher);
                        byte[] bytes = transform.TransformFinalBlock(cipherBytes, 0, cipherBytes.Length);
                        return UTF8Encoding.UTF8.GetString(bytes);
                    }
                }
            }
        }
	public async Task sendemail(string msg)  
        {  
           try  
           {  
               await Task.Run(() =>  
               {  
                   MailMessage mail = new MailMessage();  
                   SmtpClient SmtpServer = new SmtpClient("smtp.sendgrid.net");  
                   mail.From = new MailAddress("john.alamina@hud.ac.uk");  
                   mail.To.Add("u1273400@hud.ac.uk");  
                   mail.Subject = "Espnet Experiment";  
                   mail.Body = msg;  
                   //System.Net.Mail.Attachment attachment;  
		    //var loss = new System.Net.Mail.Attachment($"{root}/results/loss.png");
		    //var acc = new System.Net.Mail.Attachment($"{root}/tr_it_pytorch_train/results/acc.png");
		    //mail.Attachments.Add(acc);
		    //mail.Attachments.Add(loss);
                    SmtpServer.Port = 25;  
                   //SmtpServer.Credentials = new System.Net.NetworkCredential(ConfigurationManager.AppSettings["ApiKey"], 
					//ConfigurationManager.AppSettings["ApiKeyPass"]);  
		    SmtpServer.Credentials = new System.Net.NetworkCredential("apikey",sendgrid);
                   SmtpServer.EnableSsl = true;  
                   SmtpServer.Send(mail);  
               });  
           }  
           catch (Exception ex)  
           {  
               throw ex;  
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
	    try{
        	slackClient.Post(slackMessage);
	    }catch(Exception x){
	        Console.WriteLine($"Slack error: {x.Message}");
    	    }
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
            return tcs.Task;
        }

        private async Task tail(){
        try{
            var not_good=File.ReadAllLines($"{log}");
            var n=not_good.Length;
            //if(n>0 && msg_counter % 15 == 0)Console.WriteLine($"{n} lines.");
            if(n>0 && msg_counter % 60 == 0)Console.WriteLine(not_good[n-1]);
	    if(n>0 && msg_counter % (60 * 1) == 0) await sendemail(not_good[n-1]);
	    if(n>0 && msg_counter % (60 * 0.5) == 0)postMessage(not_good[n-1]);
	    msg_counter++;
        }catch(Exception x){
            Console.WriteLine($"{x.Message} retrying after 2s..");
            Thread.Sleep(2000);
	}
       }

	public string getTail(){
	    try{
		var line=File.ReadAllLines($"{log}");
		int n=line.Length;
		if(n>0) return line[n-1];
		return "Unable to read file..";
	    }catch(Exception x){
		Console.WriteLine($"{x.Message}");
		return x.Message;
	    }
	}

        // Define the event handlers.
        private void OnChanged(object source, FileSystemEventArgs e){
            // Specify what is done when a file is changed, created, or deleted.
            Console.WriteLine($"File: {e.FullPath} {e.ChangeType}");
            //msg_counter=0;
            //this.tail();
        }
        private void OnRenamed(object source, RenamedEventArgs e) =>
            // Specify what is done when a file is renamed.
            Console.WriteLine($"File: {e.OldFullPath} renamed to {e.FullPath}");

    }
}
