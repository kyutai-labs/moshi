// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use crate::{metrics::dispatcher as metrics, utils};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};

/// Unique identifier for sessions. These identifiers are created in order so the ordering
/// represents priority in the queue.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, PartialOrd, Ord,
)]
pub struct SessionId(u64);

#[derive(Debug, Clone)]
pub enum QueuePosition {
    Exact(usize),
    GreaterThan(usize),
}

impl SessionId {
    pub fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicU64 = atomic::AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }

    pub fn to_u64(self) -> u64 {
        self.0
    }

    pub fn from_u64(v: u64) -> Self {
        Self(v)
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct User {
    session_id: SessionId,
    queue_id: String,
    created_at: std::time::SystemTime,
    last_update: std::time::SystemTime,
    addr: Option<String>,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum RecentUserStatus {
    TimedOut,
    Queued,
    Matched,
}

struct RecentUser {
    user: User,
    status: RecentUserStatus,
    matched_instance: Option<String>,
}

// Another serialization friendly user type.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserOut {
    session_id: SessionId,
    queue_id: String,
    status: RecentUserStatus,
    matched_instance: Option<String>,
    queue_position: Option<u64>,
    created_s: f64,
    last_update_s: f64,
    addr: Option<String>,
}

// Warning: be very cautious when adding methods for Queue to ensure that `users` and
// `users_per_queue_id` stay in sync.
pub struct Queue {
    users: BTreeMap<SessionId, User>,
    users_per_queue_id: HashMap<String, usize>,
    max_queue_size: usize,
    max_recent_users: usize,
    max_connections_per_queue_id: HashMap<String, usize>,
    drop_stale_clients_after_s: f64,
    recent_users: BTreeMap<SessionId, RecentUser>,
}

impl Queue {
    pub fn new(
        max_queue_size: usize,
        max_recent_users: usize,
        max_connections_per_queue_id: HashMap<String, usize>,
        drop_stale_clients_after_s: f64,
    ) -> Self {
        Self {
            users: BTreeMap::new(),
            recent_users: BTreeMap::new(),
            max_queue_size,
            max_recent_users,
            max_connections_per_queue_id,
            users_per_queue_id: HashMap::new(),
            drop_stale_clients_after_s,
        }
    }

    // `set_config` replaces the queue limits. The new limit will only apply to incoming
    // connections.
    pub fn set_config(
        &mut self,
        max_queue_size: usize,
        max_recent_users: usize,
        max_connections_per_queue_id: HashMap<String, usize>,
        drop_stale_clients_after_s: f64,
    ) {
        self.max_connections_per_queue_id = max_connections_per_queue_id;
        self.max_queue_size = max_queue_size;
        self.max_recent_users = max_recent_users;
        self.drop_stale_clients_after_s = drop_stale_clients_after_s;
    }

    pub fn clear(&mut self) {
        self.users.clear();
        self.users_per_queue_id.clear();
    }

    pub fn len(&self) -> usize {
        self.users.len()
    }

    pub fn queue_ids(&self) -> Vec<(String, usize)> {
        self.users_per_queue_id.iter().map(|(id, cnt)| (id.to_string(), *cnt)).collect()
    }

    pub fn users(&self, include_recent_users: bool) -> Vec<UserOut> {
        let mut users = self
            .users
            .values()
            .enumerate()
            .map(|(queue_position, user)| UserOut {
                session_id: user.session_id,
                queue_id: user.queue_id.to_string(),
                created_s: utils::duration_s(user.created_at),
                last_update_s: utils::duration_s(user.last_update),
                addr: user.addr.to_owned(),
                queue_position: Some(queue_position as u64),
                status: RecentUserStatus::Queued,
                matched_instance: None,
            })
            .collect::<Vec<_>>();
        if include_recent_users {
            for ru in self.recent_users.values() {
                users.push(UserOut {
                    session_id: ru.user.session_id,
                    queue_id: ru.user.queue_id.to_string(),
                    created_s: utils::duration_s(ru.user.created_at),
                    last_update_s: utils::duration_s(ru.user.last_update),
                    addr: ru.user.addr.to_owned(),
                    queue_position: None,
                    status: ru.status,
                    matched_instance: ru.matched_instance.clone(),
                })
            }
        }
        users
    }

    #[allow(unused)]
    pub fn is_empty(&self) -> bool {
        self.users.is_empty()
    }

    pub fn add_user(&mut self, queue_id: String, addr: Option<&str>) -> Result<SessionId> {
        if self.len() >= self.max_queue_size {
            anyhow::bail!("reached max waiting queue length")
        }
        let max_users = match self.max_connections_per_queue_id.get(&queue_id) {
            None => anyhow::bail!("unknown queue id {queue_id}"),
            Some(v) => *v,
        };
        let current_users = self.users_per_queue_id.get(&queue_id).map_or(0, |v| *v);
        if current_users >= max_users {
            anyhow::bail!("max-users reached for queue id {queue_id}")
        }
        let session_id = SessionId::new();
        let now = std::time::SystemTime::now();
        let user = User {
            created_at: now,
            last_update: now,
            session_id,
            queue_id: queue_id.to_string(),
            addr: addr.map(|v| v.to_string()),
        };
        metrics::USER_IN_QUEUE.inc();
        self.users.insert(session_id, user);
        self.users_per_queue_id.insert(queue_id, current_users + 1);
        Ok(session_id)
    }

    pub fn remove_stale_users(&mut self) {
        let current_time = std::time::SystemTime::now();
        let cutoff_duration = std::time::Duration::from_secs_f64(self.drop_stale_clients_after_s);
        // TODO: use extract_if once it's stable.
        // https://doc.rust-lang.org/nightly/std/collections/struct.BTreeMap.html#method.extract_if
        let mut dropped_users = vec![];
        self.users.retain(|_key, user| {
            let since_last_update = current_time.duration_since(user.last_update);
            let retain = since_last_update.map_or(true, |v| v <= cutoff_duration);
            if !retain {
                tracing::info!(?user, "dropping stale user");
                if let Some(v) = self.users_per_queue_id.get_mut(&user.queue_id) {
                    *v = v.saturating_sub(1)
                }
                metrics::USER_TIMED_OUT.inc();
                metrics::USER_IN_QUEUE.dec();
                if let Ok(wait_time) = current_time.duration_since(user.created_at) {
                    metrics::TIMED_OUT_WAIT_TIME.observe(wait_time.as_secs_f64());
                }
                dropped_users.push(user.clone())
            }
            retain
        });
        for user in dropped_users.into_iter() {
            self.move_to_recent_users(user, RecentUserStatus::TimedOut, None)
        }
    }

    /// Returns the current position.
    pub fn refresh_user(&mut self, session_id: SessionId) -> Result<QueuePosition> {
        const MAX_POS: usize = 100;

        match self.users.get_mut(&session_id) {
            None => anyhow::bail!("unknown session_id {session_id:?}"),
            Some(user) => {
                user.last_update = std::time::SystemTime::now();
            }
        };
        let mut position = 0;
        // TODO: the tree data structure is not optimal to do this computation, hence we stop at
        // max-stop. We should probably have a specific tree that would provide node counts in log
        // time.
        for _item in self.users.range(..session_id) {
            position += 1;
            if position > MAX_POS {
                return Ok(QueuePosition::GreaterThan(MAX_POS));
            }
        }
        Ok(QueuePosition::Exact(position))
    }

    pub fn remove_on_match(&mut self, session_id: SessionId, instance_name: &str) {
        if let Some(user) = self.users.remove(&session_id) {
            let current_time = std::time::SystemTime::now();
            metrics::USER_MATCHED.inc();
            metrics::USER_IN_QUEUE.dec();
            if let Ok(wait_time) = current_time.duration_since(user.created_at) {
                metrics::MATCHED_WAIT_TIME.observe(wait_time.as_secs_f64());
            }
            if let Some(v) = self.users_per_queue_id.get_mut(&user.queue_id) {
                *v = v.saturating_sub(1)
            }
            self.move_to_recent_users(
                user,
                RecentUserStatus::Matched,
                Some(instance_name.to_string()),
            );
        }
    }

    fn move_to_recent_users(
        &mut self,
        mut user: User,
        status: RecentUserStatus,
        matched_instance: Option<String>,
    ) {
        user.last_update = std::time::SystemTime::now();
        self.recent_users.insert(user.session_id, RecentUser { user, status, matched_instance });
        while self.recent_users.len() > self.max_recent_users {
            self.recent_users.pop_first();
        }
    }
}
