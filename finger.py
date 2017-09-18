from dejavu import Dejavu
config={
	"database":{
		"host":"localhost",    
        "user":"root",       
        "passwd":"password",
        "db":"dejavu"
	}
}
djv=Dejavu(config)
djv.fingerprint_directory("/home/yash/Music/test", [".mp3",".wav"], 4)