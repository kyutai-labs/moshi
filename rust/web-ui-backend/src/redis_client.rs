use anyhow::Result;
use redis::Commands;
use std::str::FromStr;

pub struct RedisClient {
    client: redis::Client,
    key: String,
    client_map: std::collections::HashMap<String, Vec<String>>,
    client_ip: Option<std::net::Ipv4Addr>,
}

impl RedisClient {
    pub fn new(server: &str, key: &str) -> Result<Self> {
        let client = redis::Client::open(format!("redis://{}/", &server))?;
        let mut con = client.get_connection()?;
        let client_id: usize = redis::cmd("CLIENT").arg("ID").query(&mut con)?;
        let client_details: String =
            redis::cmd("CLIENT").arg("LIST").arg("ID").arg(client_id).query(&mut con)?;
        let mut client_map = std::collections::HashMap::new();
        for details in client_details.split(' ') {
            let mut details = details.split('=').map(|v| v.to_string()).collect::<Vec<_>>();
            let key = details.remove(0);
            client_map.insert(key, details);
        }
        let client_ip = client_map
            .get("addr")
            .and_then(|v| v[0].split_once(':'))
            .and_then(|(addr, _)| std::net::Ipv4Addr::from_str(addr).ok());
        Ok(Self { client, key: key.to_string(), client_map, client_ip })
    }

    pub fn client_ip(&self) -> Option<std::net::Ipv4Addr> {
        self.client_ip
    }

    pub fn client_map(&self) -> &std::collections::HashMap<String, Vec<String>> {
        &self.client_map
    }

    /// Returns the number of available workers.
    pub fn remove_stale_workers(&self) -> Result<usize> {
        let mut con = self.client.get_connection()?;
        let time = time(&mut con)?;
        con.zrembyscore(&self.key, 0., time - 60.)?;
        let available_workers: usize = con.zcount(&self.key, time, f64::INFINITY)?;
        Ok(available_workers)
    }

    /// Returns whether the key to unadvertise was removed or not.
    pub fn unadvertise(&self, v: &str) -> Result<bool> {
        let mut con = self.client.get_connection()?;
        let removed_values: usize = con.zrem(&self.key, v)?;
        Ok(removed_values > 0)
    }

    pub fn pop_worker_if_at_least(&self, min_num_workers: usize) -> Result<Option<String>> {
        let mut con = self.client.get_connection()?;
        let time = time(&mut con)?;
        let available_workers: usize = con.zcount(&self.key, time, f64::INFINITY)?;
        let worker = if available_workers < min_num_workers {
            None
        } else {
            let mut workers: Vec<(String, f64)> = con.zpopmax(&self.key, 1)?;
            if workers.is_empty() {
                None
            } else {
                let worker = workers.remove(0);
                if worker.1 <= time {
                    None
                } else {
                    Some(worker.0)
                }
            }
        };
        Ok(worker)
    }

    pub fn get_connection(&self) -> Result<redis::Connection> {
        Ok(self.client.get_connection()?)
    }

    /// ttl_us is the time to live in seconds.
    /// Returns true if the value was updated.
    pub fn advertise_worker(&self, v: &str, ttl_s: f64, update_only: bool) -> Result<bool> {
        let mut con = self.get_connection()?;
        let time = time(&mut con)?;
        // The zadd specific command cannot be used with the XX flag so we handle this manually.
        let v: redis::Value = if update_only {
            // XX + CH ensures that no new member can be inserted + the number of updated members
            // is returned.
            redis::cmd("ZADD")
                .arg(&self.key)
                .arg("XX")
                .arg("CH")
                .arg(time + ttl_s)
                .arg(v)
                .query(&mut con)?
        } else {
            con.zadd(&self.key, v, time + ttl_s)?
        };
        Ok(v != redis::Value::Nil && v != redis::Value::Int(0))
    }
}

pub fn time(con: &mut redis::Connection) -> Result<f64> {
    let (s, us): (String, String) = redis::cmd("TIME").query(con)?;
    let time = f64::from_str(&s)? + f64::from_str(&us)? / 1_000_000.0;
    Ok(time)
}
